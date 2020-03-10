#
# Parabolic error measure.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class ParabolicError(pints.ErrorMeasure):
    r"""
    Error measure based on a simple parabola centered around a user specified
    point.

    .. math::
        f(x) = (x - c)^s

    Extends :class:`pints.ErrorMeasure`.

    Parameters
    ----------
    c : sequence
        The center of the parabola.
    """
    def __init__(self, c=[0, 0]):
        self._c = pints.vector(c)
        self._n = len(self._c)

    def __call__(self, x):
        return np.sum((self._c - x)**2)

    def n_parameters(self):
        """ See :meth:`pints.ErrorMeasure.n_parameters()`. """
        return self._n

    def optimum(self):
        """
        Returns the global optimum for this function.
        """
        return np.array(self._c, copy=True)

