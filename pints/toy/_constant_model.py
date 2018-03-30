#
# Constant model with multiple outputs.
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


class ConstantModel(pints.ForwardModel):
    """
    *Extends:* :class:`pints.ForwardModel`.
    .. math::
        f(t) &= (a_1,a_2,...,a_k)

    Has a vector of parameters of dimensionality k: each of which
    is the user-specified output of a given component of the function.
    This function is mostly useful for unit testing.

    """

    def __init__(self):
        super(ConstantModel, self).__init__()
        self._no = None
        self._parameters = [0]

    def parameters(self):
        """ Returns parameters """
        return self._parameters

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return len(self._parameters)

    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.outputs()`. """
        return self._no

    def simulate(self, parameters, times):
        return self._simulate(parameters, times)

    def _simulate(self, parameters, times):
        self._no = len(parameters)
        self._parameters = [float(x) for x in parameters]
        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')
        if len(parameters) < 1:
            raise ValueError('Function takes at least 1 parameter')
        if np.any(np.isnan(self._parameters)):
            raise ValueError('Parameters must be a number.')
        if np.any(np.isinf(self._parameters)):
            raise ValueError('Parameters must be finite.')

        if self._no == 1:
            return self._parameters[0] * np.ones(times.shape)
        else:
            return np.transpose(np.asarray([x * np.ones(times.shape)
                                            for x in self._parameters]))
