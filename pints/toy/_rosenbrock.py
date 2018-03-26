#
# Rosenbrock error measure and log-pdf
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class RosenbrockError(pints.ErrorMeasure):
    """
    *Extends:* :class:`pints.ErrorMeasure`.

    Error measure based on the rosenbrock function (see:
    https://en.wikipedia.org/wiki/Rosenbrock_function):

    .. math::
        f(x,y) = (a - x)^2 + b(y - x^2)^2

    """
    def __init__(self, a=1, b=100):
        self._a = float(a)
        self._b = float(b)

    def dimension(self):
        """ See :meth:`pints.ErrorMeasure.dimension()`. """
        return 2

    def optimum(self):
        """
        Returns the global optimum for this function.
        """
        return self._a, self._a**2

    def __call__(self, x):
        return (self._a - x[0])**2 + self._b * (x[1] - x[0]**2)**2


class RosenbrockLogPDF(pints.LogPDF):
    """
    *Extends:* :class:`pints.LogPDF`.

    Unnormalised LogPDF based on the Rosenbrock function (see:
    https://en.wikipedia.org/wiki/Rosenbrock_function):

    .. math::
        f(x,y) = -log[ (a - x)^2 + b(y - x^2)^2 ]

    """
    def __init__(self, a=1, b=100):
        self._f = RosenbrockError(a, b)

    def dimension(self):
        """ See :meth:`pints.LogPDF.dimension()`. """
        return self._f.dimension()

    def optimum(self):
        """
        Returns the global optimum for this log-pdf.
        """
        return self._f.optimum()

    def __call__(self, x):
        f = self._f(x)
        return float('inf') if f == 0 else -np.log(f)

