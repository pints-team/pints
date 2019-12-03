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

from . import ToyModel


class StochasticLogisticModel(pints.ForwardModel, ToyModel):
    r"""

    *Extends:* :class:`pints.ForwardModel`, :class:`pints.toy.ToyModel`.
    """

    def __init__(self, initial_molecule_count=2):
        super(StochasticLogisticModel, self).__init__()
        self._n0 = float(initial_molecule_count)
        if self._n0 < 0:
            raise ValueError('Initial molecule count cannot be negative.')

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return 2

    def simulate_raw(self, parameters):
        """
        Returns raw times, population sizes when reactions occur
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

    def interpolate_values(self, time, pop_size, output_times, parameters):
        """
        Takes raw times and population size values as inputs
        and outputs interpolated values at output_times
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
        time, pop_size = self.simulate_raw(parameters)

        # interpolate
        values = self.interpolate_values(time, pop_size, times, parameters)
        return values

    def mean(self, parameters, times):
        r"""
        Returns the deterministic mean of infinitely many stochastic
        simulations, which follows:
        :math:`\frac{kC(0)}{C(0) + \exp{-kt}(k - C(0))}`.
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
        raise NotImplementedError()

    def suggested_parameters(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """
        return np.array([0.1, 50])

    def suggested_times(self):
        """ See "meth:`pints.toy.ToyModel.suggested_times()`."""
        return np.linspace(0, 100, 101)
