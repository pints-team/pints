#
# Logistic toy model.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import pints

from . import ToyModel


class StochasticDegradationModel(pints.ForwardModel, ToyModel):

    """
    Stochastic decay model of a single chemical reaction [1].

    Time until next reaction...
    .. math::
    $\tau = \frac{1}{A(t)k}*\ln{\frac{1}{r}}$

    Has one parameter: Rate constant :math:`k`.
    :math:`r` is a random variable, which is part of the stochastic model
    The initial concentration :math:`A(0) = n_0` can be set using the
    (optional) named constructor arg ``initial_concentration``

    [1] Erban et al., 2007

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
            raise ValueError('rate constant must be positive')

        k = [float(parameters)]
        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')
        if self._n0 == 0:
            return np.zeros(times.shape)
        a = self._n0

        t = 0
        mol_conc = [a]
        time = [t]

        # Run stochastic degradation algorithm, calculating time until next
        # reaction and decreasing concentration by 1 at that time
        while a > 0:
            r = np.random.uniform(0, 1)
            tao = (1 / (a * k)) * np.log(1 / r)
            t += tao
            time.append(t)
            a = a - 1
            mol_conc.append(a)

        # Interpolate as step function, decreasing mol_conc by 1 at each
        # reaction time point
        f1 = interp1d(time, mol_conc, kind='previous')

        # Compute concentration ('a') values at given time points using f1
        # at any time beyond the last reaction, concentration = 0
        values = f1(times[np.where(times <= max(time))])
        zero_vector = np.zeros(len(times[np.where(times > max(time))]))
        values = np.concatenate((values, zero_vector))

        return values

    def suggested_parameter(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """
        return 0.1

