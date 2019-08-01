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
    an initial concentration n_0 and degrading to 0 according to the following
    model:
    .. math::
    $A rightarrow{\text{k}} 0 $ [1]

    The model is simulated according to the Gillespie algorithm [2]:
    1. Sample a random value r from a uniform distribution: r ~ unif(0,1)
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

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return 1

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        if parameters <= 0:
            raise ValueError('Rate constant must be positive.')

        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')
        if self._n0 == 0:
            return np.zeros(times.shape)

        [time, mol_count] = self.StochasticSimulationAlgorithm(parameters)

        # Interpolate as step function, decreasing mol_count by 1 at each
        # reaction time point
        f1 = interp1d(time, mol_count, kind='previous')

        # Compute concentration ('a') values at given time points using f1
        # at any time beyond the last reaction, concentration = 0
        values = f1(times[np.where(times <= max(time))])
        zero_vector = np.zeros(len(times[np.where(times > max(time))]))
        values = np.concatenate((values, zero_vector))

        return values

    def StochasticSimulationAlgorithm(self, parameters):
        t = 0
        if parameters <= 0:
            raise ValueError('Rate constant must be positive.')
        k = np.array([float(parameters)])

        a = np.array([float(self._n0)])
        mol_count = [a[0]]
        time = [t]

        # Run stochastic degradation algorithm, calculating time until next
        # reaction and decreasing concentration by 1 at that time
        while a[0] > 0:
            r = np.random.uniform(0, 1)
            tao = ((1 / (a * k)) * np.log(1 / r))[0]
            t += tao
            time.append(t)
            a[0] = a[0] - 1
            mol_count.append(a[0])

        return time, mol_count

    def DeterministicMean(self, parameters, times):
        if parameters <= 0:
            raise ValueError('Rate constant must be positive.')
        k = parameters

        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')

        mean = self._n0*np.exp(-k*times)

        return mean

    def suggested_parameter(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """
        return 0.1

    def suggested_times(self):
        """ See "meth:`pints.toy.ToyModel.suggested_times()`."""
        return np.linspace(0, 100, 101)

