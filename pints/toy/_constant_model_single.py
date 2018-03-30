#
# Constant model with single output.
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


class ConstantModelSingle(pints.ForwardModel):
    """
    *Extends:* :class:`pints.ForwardModel`.
    .. math::
        f(t) &= a

    Has one parameter: :math:`r`, which is the user-specified
    output of the function. This function is mostly useful for
    unit testing.

    """

    def __init__(self):
        super(ConstantModelSingle, self).__init__()
        self._a = -99

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return 1

    def simulate(self, parameters, times):
        return self._simulate(parameters, times)

    def _simulate(self, parameters, times):
        self._a = float(parameters[0])
        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')
        if len(parameters) != 1:
            raise ValueError('Function takes a single parameter')
        if np.isnan(self._a):
            raise ValueError('Parameter must be a number.')
        if np.isinf(self._a):
            raise ValueError('Parameter must be finite.')
        return self._a * np.ones(times.shape)
