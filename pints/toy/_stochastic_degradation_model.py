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


class StochasticDegradationModel(pints.ForwardModel, ToyModel):

    """
    Stochastic degradation model of a single chemical reaction starting from
    an initial concentration :math:: n0 and degrading to 0 according to the
    following model:
    .. math::
    $A rightarrow{\text{k}} 0 $ [1]

    The model is simulated according to the Gillespie algorithm [2]:
    1. Sample a random value r from a uniform distribution:
    :math:: r ~ unif(0,1)
    2. Calculate the time ($\tau$) until the next single reaction as follows:
       .. math::
       $\tau = \frac{1}{A(t)k}*ln{\frac{1}{r}}$ [1]
    3. Update the molecule count at time t + .. math:: $\tau$ as:
    .. math:: $A(t + \tau) = A(t)-1$
    4. Return to step (1) until molecule count reaches 0

    Has one parameter: Rate constant :math:`k`.
    :math:`r` is a random variable, which is part of the stochastic model
    The initial concentration :math:`A(0) = n_0` can be set using the
    (optional) named constructor arg ``initial_concentration``

    [1] Erban et al., 2007
    [2] Gillespie, 1976

    *Extends:* :class:`pints.ForwardModel`, :class:`pints.toy.ToyModel`.
    """

    def __init__(self, initial_concentration=20):
        super(StochasticDegradationModel, self).__init__()
        self._n0 = float(initial_concentration)
        if self._n0 < 0:
            raise ValueError('Initial concentration cannot be negative.')

        self._interp_func = None
        self._mol_count = []
        self._time = []

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return 1

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        parameters = np.asarray(parameters)
        if len(parameters) != self.n_parameters():
            raise ValueError('This model should have only 1 parameter.')
        k = parameters[0]

        if k <= 0:
            raise ValueError('Rate constant must be positive.')

        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')
        if self._n0 == 0:
            return np.zeros(times.shape)

        t = 0
        a = self._n0
        self._mol_count = [a]
        self._time = [t]

        # Run stochastic degradation algorithm, calculating time until next
        # reaction and decreasing concentration by 1 at that time
        while a > 0:
            r = np.random.uniform(0, 1)
            t += (1 / (a * k)) * np.log(1 / r)
            self._time.append(t)
            a = a - 1
            self._mol_count.append(a)

        # Interpolate as step function, decreasing mol_count by 1 at each
        # reaction time point
        self._interp_func = interp1d(self._time, self._mol_count,
                                     kind='previous')

        # Compute concentration values at given time points using f1
        # at any time beyond the last reaction, concentration = 0
        values = self._interp_func(times[np.where(times <= max(self._time))])
        zero_vector = np.zeros(len(times[np.where(times > max(self._time))]))
        values = np.concatenate((values, zero_vector))

        return values

    def deterministic_mean(self, parameters, times):
        """ Calculates deterministic mean of infinitely many stochastic
        simulations, which follows :math:: n0*exp(-kt)"""
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

    def deterministic_variance(self, parameters, times):
        """ Calculates deterministic variance of infinitely many stochastic
        simulations, which follows :math:: exp(-2kt)(-1 + exp(kt)) * n0"""
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

