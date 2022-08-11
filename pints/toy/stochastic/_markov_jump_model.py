#
# Markov jump model.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
import scipy.interpolate

import pints

from .. import ToyModel


class MarkovJumpModel(pints.ForwardModel, ToyModel):
    r"""
    A general purpose Markov Jump model used for any systems of reactions
    that proceed through jumps.

    A population of N different species is simulated, reacting through M
    different reaction equations.

    Simulations are performed using Gillespie's algorithm [1]_, [2]_:

    1. Sample values :math:`r_0`, :math:`r_1`, from a uniform distribution

    .. math::
        r_0, r_1 \sim U(0,1)

    2. Calculate the time :math:`\tau` until the next single reaction as

    .. math::
        \tau = \frac{-\ln(r)}{a_0}

    where :math:`a_0` is the sum of the propensities at the current time.

    3. Decide which reaction, i, takes place using :math:`r_1 * a_0` and
    iterating through propensities. Since :math:`r_1` is a a value between 0
    and 1 and :math`a_0` is the sum of all propensities, we can find :math:`k`
    for which :math:`s_k / a_0 <= r_2 < s_(k+1) / a_0` where :math:`s_j` is the
    sum of the first :math:`j` propensities at time :math:`t`. We then choose
    :math:`i` as the reaction corresponding to propensity k.

    4. Update the state :math:`x` at time :math:`t + \tau` as:

    .. math::
        x(t + \tau) = x(t) + V[i]

    4. Return to step (1) until no reaction can take place or the process
    has gone past the maximum time.

    Extends :class:`pints.ForwardModel`, :class:`pints.toy.ToyModel`.

    Parameters
    ----------
    x_0
        An N-vector specifying the initial population of each
        of the N species.
    V
        An NxM matrix consisting of stochiometric vectors :math:`v_i`
        specifying the changes to the state, x, from reaction i taking place.
    propensities
        A function from the current state, x, and reaction rates, k, to a
        vector of the rates of each reaction taking place.

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
    def __init__(self, x0, V, propensities):
        super(MarkovJumpModel, self).__init__()
        self._x0 = np.asarray(x0)
        self._V = V
        self._propensities = propensities
        if any(self._x0 < 0):
            raise ValueError('Initial molecule count cannot be negative.')

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return len(self._V)

    def simulate_raw(self, rates, max_time):
        """ Returns raw times, mol counts when reactions occur. """
        if len(rates) != self.n_parameters():
            raise ValueError(
                'This model should have ' + str(self.n_parameters())
                + ' parameter(s).')

        # Setting the current propensities and summing them up
        current_propensities = self._propensities(self._x0, rates)
        prop_sum = np.sum(current_propensities)

        # Initial time and count
        t = 0
        x = np.array(self._x0)

        # Run Gillespie SSA, calculating time until next reaction, deciding
        # which reaction, and applying it
        mol_count = [np.array(x)]
        time = [t]
        while prop_sum > 0 and t <= max_time:
            r_1, r_2 = np.random.uniform(0, 1), np.random.uniform(0, 1)
            t += -np.log(r_1) / prop_sum
            s = 0
            r = 0
            while s <= r_2 * prop_sum:
                s += current_propensities[r]
                r += 1
            x += self._V[r - 1]

            # Calculate new current propensities
            current_propensities = self._propensities(x, rates)
            prop_sum = np.sum(current_propensities)

            # Store new values
            time.append(t)
            mol_count.append(np.copy(x))

        return np.array(time), np.array(mol_count)

    def interpolate_mol_counts(self, time, mol_count, output_times):
        """
        Takes raw times and inputs and mol counts and outputs interpolated
        values at output_times
        """
        if len(time) == 0:
            raise ValueError('At least one time must be given.')
        if len(time) != len(mol_count):
            raise ValueError(
                'The number of entries in time must match mol_count')

        # Check output times
        output_times = np.asarray(output_times)
        if not np.all(output_times[1:] >= output_times[:-1]):
            raise ValueError('The output_times must be non-decreasing.')

        # Interpolate as step function, decreasing mol_count by 1 at each
        # reaction time point.
        if len(time) == 1:
            # Need at least 2 values to interpolate
            return np.ones(len(output_times)) * mol_count[0]
        else:
            # Note: Can't use fill_value='extrapolate' here as:
            #  1. This require scipy >= 0.17
            #  2. There seems to be a bug in some scipy versions
            interp_func = scipy.interpolate.interp1d(
                time, mol_count, kind='previous', axis=0, fill_value=np.nan,
                bounds_error=False)
            values = interp_func(output_times)

        # At any point past the final time, repeat the last value
        values[output_times >= time[-1]] = mol_count[-1]
        values[output_times < time[0]] = mol_count[0]

        return values

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')

        # Run Gillespie algorithm
        time, mol_count = self.simulate_raw(parameters, max(times))

        # Interpolate and return
        return self.interpolate_mol_counts(time, mol_count, times)

