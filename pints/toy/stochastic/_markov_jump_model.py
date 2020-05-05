#
# Stochastic degradation toy model.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
from scipy.interpolate import interp1d
import pints
import random

from .. import ToyModel


class MarkovJumpModel(pints.ForwardModel, ToyModel):
    r"""
    A general purpose Markov Jump model used for any systems of reactions
    that proceed through jumps. We simulate a population of N different species
    reacting through M different mechanisms.

    A model has three parameters:
        - x_0 - an N-vector specifying the initial population of each
            of the N species
        - V - an NxM matrix consisting of stochiometric vectors v_i specifying
            the changes to the state, x,  from reaction i taking place
        - a - a function from the current state, x, and reaction rates, k,
            to a vector of the rates of each reaction taking place

    Simulations are performed using Gillespie's algorithm [1]_, [2]_:

    1. Sample values :math:`r_0`, :math:`r_1`, from a uniform distribution

    .. math::
        r_0, r_1 \sim U(0,1)

    2. Calculate the time :math:`\tau` until the next single reaction as

    .. math::
        \tau = \frac{-\ln(r)}{a_0}

    3. Decide which reaction, i, takes place using r_1 * a_0 and iterating
    through propensities

    4. Update the state :math:`x` at time :math:`t + \tau` as:

    .. math::
        x(t + \tau) = x(t) + V[i]

    4. Return to step (1) until no reaction can take place

    The model has one parameter, the rate constant :math:`k`.

    Extends :class:`pints.ForwardModel`, :class:`pints.toy.ToyModel`.

    Parameters
    ----------
    initial_molecule_count
        The initial molecule count :math:`A(0)`.

    References
    ----------
    .. [1] A Practical Guide to Stochastic Simulations of Reaction Diffusion
           Processes. Erban, Chapman, Maini (2007).
           arXiv:0704.1908v2 [q-bio.SC]
           https://arxiv.org/abs/0704.1908

    .. [2] A general method for numerically simulating the stochastic time
           evolution of coupled chemical reactions. Gillespie (1976).
           Journal of Computational Physics
           https://doi.org/10.1016/0021-9991(76)90041-3
    """
    def __init__(self, x0, V, a):
        super(MarkovJumpModel, self).__init__()
        self._x0 = np.asarray(x0)
        self._V = V
        self._a = a
        if any(self._x0 < 0):
            raise ValueError('Initial molecule count cannot be negative.')

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return len(self._V)

    def simulate_raw(self, rates, max_time):
        """
        Returns raw times, mol counts when reactions occur
        """
        # parameters = np.asarray(parameters)
        # if len(parameters) != self.n_parameters():
        #     raise ValueError('This model should have only 1 parameter.')
        # k = parameters[0]

        current_rates = self._a(self._x0, rates)
        a_0 = sum(current_rates)

        # Initial time and count
        t = 0
        x = np.array(self._x0)

        # Run gillespie SSA, calculating time until next
        # reaction, deciding which reaction, and applying it
        mol_count = [np.array(x)]
        time = [t]
        while a_0 > 0 and t <= max_time:
            r_1, r_2 = random.random(), random.random()
            t += -np.log(r_1) / (a_0)
            s = 0
            r = 0
            while s <= r_2 * a_0:
                s += current_rates[r]
                r += 1
            r -= 1
            x = np.add(x, self._V[r])

            current_rates = self._a(x, rates)
            a_0 = sum(current_rates)

            time.append(t)
            mol_count.append(np.array(x))
        return time, mol_count

    def simulate_approx(self, rates, max_time, tau):
        assert tau > 0, "cannot tau-leap with negative tau"
        current_rates = np.array(self._a(self._x0, rates))
        # Initial time and count
        t = 0
        x = self._x0.copy()
        N = len(rates)
        # Run gillespie SSA, calculating time until next
        # reaction, deciding which reaction, and applying it
        mol_count = [x.copy()]
        time = [t]
        while any(current_rates > 0) and t <= max_time:
            # Estimate number of each reaction in [t, t+tau)
            k = [np.random.poisson(current_rates[i] * tau) for i in range(N)]

            # Apply the reactions
            for r in range(N):
                x += np.array(self._V[r]) * k[r]

            # Update rates
            current_rates = np.array(self._a(x, rates))

            # Advance Time
            t += tau
            time.append(t)
            mol_count.append(x.copy())
        return time, mol_count

    def interpolate_mol_counts(self, time, mol_count, output_times):
        """
        Takes raw times and inputs and mol counts and outputs interpolated
        values at output_times
        """
        # Interpolate as step function, decreasing mol_count by 1 at each
        # reaction time point
        interp_func = interp1d(time, mol_count, kind='previous', axis=0,
                               fill_value="extrapolate", bounds_error=False)

        # Compute molecule count values at given time points using f1
        # at any time beyond the last reaction, molecule count = 0
        values = interp_func(output_times)
        return values

    def simulate(self, parameters, times, approx_tau=None):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')
        if approx_tau is None:
            # run Gillespie
            time, mol_count = self.simulate_raw(parameters, max(times))
        else:
            if (not approx_tau) or approx_tau <= 0:
                ValueError("You must provide a positive value for approx_tau\
                 to use tau-leaping approximation")
            # Run Euler tau-leaping
            time, mol_count = self.simulate_approx(parameters, max(times),
                                                   approx_tau)
        # interpolate
        values = self.interpolate_mol_counts(np.asarray(time),
                                             np.asarray(mol_count), times)
        return values

