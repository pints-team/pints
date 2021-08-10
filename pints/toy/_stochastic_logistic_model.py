#
# Stochastic logistic model.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
from scipy.interpolate import interp1d
import pints

from . import ToyModel


class StochasticLogisticModel(pints.ForwardModel, ToyModel):
    r"""
    This model describes the growth of a population of individuals, where the
    birth rate per capita, initially :math:`b_0`, decreases to :math:`0` as the
    population size, :math:`\mathcal{C}(t)`, starting from an initial
    population size, :math:`n_0`, approaches a carrying capacity, :math:`k`.
    This process follows a rate according to [1]_

    .. math::
       A \xrightarrow{b_0(1-\frac{\mathcal{C}(t)}{k})} 2A.

    The model is simulated using the Gillespie stochastic simulation algorithm
    [2]_, [3]_.

    *Extends:* :class:`pints.ForwardModel`, :class:`pints.toy.ToyModel`.

    Parameters
    ----------
    initial_molecule_count : float
        Sets the initial population size :math:`n_0`.

    References
    ----------
    .. [1] Simpson, M. et al. 2019. Process noise distinguishes between
           indistinguishable population dynamics. bioRxiv.
           https://doi.org/10.1101/533182
    .. [2] Gillespie, D. 1976. A General Method for Numerically Simulating the
           Stochastic Time Evolution of Coupled Chemical Reactions.
           Journal of Computational Physics. 22 (4): 403-434.
           https://doi.org/10.1016/0021-9991(76)90041-3
    .. [3] Erban R. et al. 2007. A practical guide to stochastic simulations
           of reaction-diffusion processes. arXiv.
           https://arxiv.org/abs/0704.1908v2
    """

    def __init__(self, initial_molecule_count=50):
        super(StochasticLogisticModel, self).__init__()
        self._n0 = float(initial_molecule_count)
        if self._n0 < 0:
            raise ValueError('Initial molecule count cannot be negative.')

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return 2

    def _simulate_raw(self, parameters):
        """
        Returns tuple (raw times, population sizes) when reactions occur.
        """
        parameters = np.asarray(parameters)
        if len(parameters) != self.n_parameters():
            raise ValueError('This model should have only 2 parameters.')
        b = parameters[0]
        k = parameters[1]
        if b <= 0:
            raise ValueError('Rate constant must be positive.')

        # Initial time and count
        t = 0
        a = self._n0

        # Run stochastic logistic birth-only algorithm, calculating time until
        # next reaction and increasing population count by 1 at that time
        mol_count = [a]
        time = [t]
        while a < k:
            r = np.random.uniform(0, 1)
            t += np.log(1 / r) / (a * b * (1 - a / k))
            a = a + 1
            time.append(t)
            mol_count.append(a)
        return time, mol_count

    def _interpolate_values(self, time, pop_size, output_times, parameters):
        """
        Takes raw times and population size values as inputs and outputs
        interpolated values at output_times.
        """
        # Interpolate as step function, increasing pop_size by 1 at each
        # event time point
        interp_func = interp1d(time, pop_size, kind='previous')

        # Compute population size values at given time points using f1
        # at any time beyond the last event, pop_size = k
        values = interp_func(output_times[np.where(output_times <= time[-1])])
        zero_vector = np.full(
            len(output_times[np.where(output_times > time[-1])]),
            parameters[1])
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
        time, pop_size = self._simulate_raw(parameters)

        # interpolate
        values = self._interpolate_values(time, pop_size, times, parameters)
        return values

    def mean(self, parameters, times):
        r"""
        Computes the deterministic mean of infinitely many stochastic
        simulations with times :math:`t` and parameters (:math:`b`, :math:`k`),
        which follows:
        :math:`\frac{kC(0)}{C(0) + (k - C(0)) \exp(-bt)}`.

        Returns an array with the same length as `times`.
        """
        parameters = np.asarray(parameters)
        if len(parameters) != self.n_parameters():
            raise ValueError('This model should have only 2 parameters.')

        b = parameters[0]
        if b <= 0:
            raise ValueError('Rate constant must be positive.')

        k = parameters[1]
        if k <= 0:
            raise ValueError("Carrying capacity must be positive")

        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')
        c0 = self._n0
        return (c0 * k) / (c0 + np.exp(-b * times) * (k - c0))

    def variance(self, parameters, times):
        r"""
        Returns the deterministic variance of infinitely many stochastic
        simulations.
        """
        raise NotImplementedError

    def suggested_parameters(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """
        return np.array([0.1, 500])

    def suggested_times(self):
        """ See :meth:`pints.toy.ToyModel.suggested_times()`."""
        return np.linspace(0, 100, 101)
