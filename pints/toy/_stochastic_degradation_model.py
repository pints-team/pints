#
# Stochastic degradation toy model.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
from scipy.interpolate import interp1d
import pints

from . import ToyModel


class StochasticDegradationModel(pints.ForwardModel, ToyModel):
    r"""
    Stochastic degradation model of a single chemical reaction starting from
    an initial molecule count :math:`A(0)` and degrading to 0 with a fixed rate
    :math:`k`:

    .. math::
        A \xrightarrow{k} 0

    Simulations are performed using Gillespie's algorithm [1]_, [2]_:

    1. Sample a random value :math:`r` from a uniform distribution

    .. math::
        r \sim U(0,1)

    2. Calculate the time :math:`\tau` until the next single reaction as

    .. math::
        \tau = \frac{-\ln(r)}{A(t) k}

    3. Update the molecule count :math:`A` at time :math:`t + \tau` as:

    .. math::
        A(t + \tau) = A(t) - 1

    4. Return to step (1) until the molecule count reaches 0

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
    def __init__(self, initial_molecule_count=20):
        super(StochasticDegradationModel, self).__init__()
        self._n0 = float(initial_molecule_count)
        if self._n0 < 0:
            raise ValueError('Initial molecule count cannot be negative.')

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return 1

    def simulate_raw(self, parameters):
        """
        Returns raw times, mol counts when reactions occur
        """
        parameters = np.asarray(parameters)
        if len(parameters) != self.n_parameters():
            raise ValueError('This model should have only 1 parameter.')
        k = parameters[0]

        if k <= 0:
            raise ValueError('Rate constant must be positive.')

        # Initial time and count
        t = 0
        a = self._n0

        # Run stochastic degradation algorithm, calculating time until next
        # reaction and decreasing molecule count by 1 at that time
        mol_count = [a]
        time = [t]
        while a > 0:
            r = np.random.uniform(0, 1)
            t += -np.log(r) / (a * k)
            a = a - 1
            time.append(t)
            mol_count.append(a)
        return time, mol_count

    def interpolate_mol_counts(self, time, mol_count, output_times):
        """
        Takes raw times and inputs and mol counts and outputs interpolated
        values at output_times
        """
        # Interpolate as step function, decreasing mol_count by 1 at each
        # reaction time point
        interp_func = interp1d(time, mol_count, kind='previous')

        # Compute molecule count values at given time points using f1
        # at any time beyond the last reaction, molecule count = 0
        values = interp_func(output_times[np.where(output_times <= time[-1])])
        zero_vector = np.zeros(
            len(output_times[np.where(output_times > time[-1])])
        )
        values = np.concatenate((values, zero_vector))
        return values

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')
        if self._n0 == 0:
            return np.zeros(times.shape)

        # run Gillespie
        time, mol_count = self.simulate_raw(parameters)

        # interpolate
        values = self.interpolate_mol_counts(time, mol_count, times)
        return values

    def mean(self, parameters, times):
        r"""
        Returns the deterministic mean of infinitely many stochastic
        simulations, which follows :math:`A(0) \exp(-kt)`.
        """
        parameters = np.asarray(parameters)
        if len(parameters) != self.n_parameters():
            raise ValueError('This model should have only 1 parameter.')
        k = parameters[0]

        if k <= 0:
            raise ValueError('Rate constant must be positive.')

        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')

        mean = self._n0 * np.exp(-k * times)
        return mean

    def variance(self, parameters, times):
        r"""
        Returns the deterministic variance of infinitely many stochastic
        simulations, which follows :math:`\exp(-2kt)(-1 + \exp(kt))A(0)`.
        """
        parameters = np.asarray(parameters)
        if len(parameters) != self.n_parameters():
            raise ValueError('This model should have only 1 parameter.')
        k = parameters[0]

        if k <= 0:
            raise ValueError('Rate constant must be positive.')

        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')

        variance = np.exp(-2 * k * times) * (-1 + np.exp(k * times)) * self._n0
        return variance

    def suggested_parameters(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """
        return np.array([0.1])

    def suggested_times(self):
        """ See "meth:`pints.toy.ToyModel.suggested_times()`."""
        return np.linspace(0, 100, 101)
