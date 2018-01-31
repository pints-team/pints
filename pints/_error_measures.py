#
# Scoring functions
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


class ErrorMeasure(object):
    """
    Abstract base class.

    Calculates some scalar measure of goodness-of-fit for a model and a data
    set, such that a smaller value means a better fit.
    """
    def __init__(self, problem=None, dimension=None):
        if problem:
            self._problem = problem
            self._times = problem.times()
            self._values = problem.values()
            self._dimension = problem.dimension()
        else:
            self._dimension = float(dimension)

    def __call__(self, x):
        raise NotImplementedError

    def dimension(self):
        """
        Returns the dimension of the space this measure is defined on.
        """
        return self._dimension


class ProbabilityBasedError(ErrorMeasure):
    """
    *Extends:* :class:`ErrorMeasure`

    Changes the sign of a :class:`LogPDF` to use it as an error.
    """
    def __init__(self, log_pdf):
        if not isinstance(log_pdf, pints.LogPDF):
            raise ValueError(
                'Given log_pdf must be an instance of pints.LogPDF.')
        self._log_pdf = log_pdf
        super(ProbabilityBasedError, self).__init__(
            dimension=log_pdf.dimension())

    def __call__(self, x):
        return -self._log_pdf(x)


class RMSError(ErrorMeasure):
    """
    *Extends:* :class:`ErrorMeasure`

    Calculates the square root of a normalised sum-of-squares error:
    ``f = sqrt( sum( (x[i] - y[i])**2 / n) )``
    """
    def __init__(self, problem):
        super(RMSError, self).__init__(problem)
        self._ninv = 1.0 / len(self._values)

    def __call__(self, x):
        return np.sqrt(self._ninv * np.sum(
            (self._problem.evaluate(x) - self._values)**2))


class SumOfSquaresError(ErrorMeasure):
    """
    *Extends:* :class:`ErrorMeasure`

    Calculates a sum-of-squares error: ``f = sum( (x[i] - y[i])**2 )``
    """
    def __call__(self, x):
        return np.sum((self._problem.evaluate(x) - self._values)**2)

