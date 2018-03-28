#
# Logistic model.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import pints


class LogisticModel(pints.ForwardModel):
    """
    Logistic model.

    .. math::
        f(t) &= \\frac{k}{1+(k/p_0 - 1)*\exp(-r t)} \\\\
        \\frac{df(t)}{dr} &= \\frac{k t (k / p_0 - 1) \exp(-r t)}
                                   {((k/p_0-1) \exp(-r t) + 1)^2} \\\\
        \\frac{df(t)}{dk} &= \\frac{k \exp(-r t)}
                                   {p_0 ((k/p_0-1)\exp(-r t) + 1)^2} 
                             + \\frac{1}{(k/p_0 - 1)\exp(-r t) + 1}

    Has two parameters: A growth rate :math:`r` and a carrying capacity 
    :math:`k`. The initial population size :math:`f(0) = p_0` can be set using
    the (optional) named constructor arg ``initial_population_size``
    """

    def __init__(self, initial_population_size=2):
        super(LogisticModel, self).__init__()
        self._p0 = float(initial_population_size)
        if self._p0 < 0:
            raise ValueError('Population size cannot be negative.')

    def dimension(self):
        return 2

    def simulate(self, parameters, times):
        r, k = [float(x) for x in parameters]
        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')
        if self._p0 == 0:
            return np.zeros(times.shape)
        if k < 0:
            return np.zeros(times.shape)
        return k / (1 + (k / self._p0 - 1) * np.exp(-r * times))

    def sensitivities(self, parameters, times):
        r, k = [float(x) for x in parameters]
        t = np.asarray(times)
        result = np.empty((len(times), len(parameters)))
        exp = np.exp(-r * t)
        c = (k / self._p0 - 1)
        result[:, 0] = k * t * c * exp / (c * exp + 1)**2
        result[:, 1] = -k * exp / \
            (self._p0 * (c * exp + 1)**2) + 1 / (c * exp + 1)
        return result
