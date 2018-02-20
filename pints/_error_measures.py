#
# Scoring functions
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class ErrorMeasure(object):
    """
    Abstract base class for objects that calculate some scalar measure of
    goodness-of-fit (for a model and a data set), such that a smaller value
    means a better fit.

    ErrorMeasures are callable objects: If ``e`` is an instance of an
    ``ErrorMeasure`` class you can calculate the error by calling ``e(p)``
    where ``p`` is a point in parameter space.
    """
    def __call__(self, x):
        raise NotImplementedError

    def dimension(self):
        raise NotImplementedError


class ProblemErrorMeasure(ErrorMeasure):
    """
    Abstract base class for ErrorMeasures defined for
    :class:`Problems <pints.Problem>`.
    """
    def __init__(self, problem=None):
        super(ProblemErrorMeasure, self).__init__()
        self._problem = problem
        self._times = problem.times()
        self._values = problem.values()
        self._dimension = problem.dimension()

    def dimension(self):
        """ See :meth:`ErrorMeasure.dimension`. """
        return self._dimension


class ProbabilityBasedError(ErrorMeasure):
    """
    *Extends:* :class:`ErrorMeasure`

    Changes the sign of a :class:`LogPDF` to use it as an error. Minimising
    this error will maximise the probability.
    """
    def __init__(self, log_pdf):
        super(ProbabilityBasedError, self).__init__()
        if not isinstance(log_pdf, pints.LogPDF):
            raise ValueError(
                'Given log_pdf must be an instance of pints.LogPDF.')
        self._log_pdf = log_pdf

    def dimension(self):
        """ See :meth:`ErrorMeasure.dimension`. """
        return self._log_pdf.dimension()

    def __call__(self, x):
        return -self._log_pdf(x)


class SumOfErrors(ErrorMeasure):
    """
    *Extends:* :class:`ErrorMeasure`

    Calculates a sum of :class:`ErrorMeasure` objects, all defined on the same
    parameter space.
    """
    def __init__(self, *error_measures):
        super(SumOfErrors, self).__init__()

        # Check input arguments
        if len(error_measures) < 2:
            raise ValueError('SumOfErrors requires at least 2 error measures.')
        for i, e in enumerate(error_measures):
            if not isinstance(e, pints.ErrorMeasure):
                raise ValueError(
                    'All objects passed to SumOfErrors must be instances of'
                    ' pints.ErrorMeasure (failed on argument ' + str(i) + '.')
        self._errors = error_measures

        # Get and check dimension
        i = iter(self._errors)
        self._dimension = next(i).dimension()
        for e in i:
            if e.dimension() != self._dimension:
                raise ValueError(
                    'All errors passed to sum of errors must have same'
                    ' dimension.')

    def dimension(self):
        """ See :meth:`ErrorMeasure.dimension`. """
        return self._dimension

    def __call__(self, x):
        total = 0
        for e in self._errors:
            total += e(x)
            if not np.isfinite(total):
                return total
        return total



class MeanSquaredError(ProblemErrorMeasure):
    """
    *Extends:* :class:`ProblemErrorMeasure`

    Calculates the mean square error: ``f = sum( (x[i] - y[i])**2 ) / n``
    """
    def __init__(self, problem):
        super(MeanSquaredError, self).__init__(problem)
        self._ninv = 1.0 / len(self._values)

    def __call__(self, x):
        return (np.sum((self._problem.evaluate(x) - self._values)**2) *
                self._ninv)


class RootMeanSquaredError(ProblemErrorMeasure):
    """
    *Extends:* :class:`ProblemErrorMeasure`

    Calculates a root mean squared error (RMSE):
    ``f = sqrt( sum( (x[i] - y[i])**2 / n) )``
    """
    def __init__(self, problem):
        super(RootMeanSquaredError, self).__init__(problem)
        self._ninv = 1.0 / len(self._values)

    def __call__(self, x):
        return np.sqrt(self._ninv * np.sum(
            (self._problem.evaluate(x) - self._values)**2))


class SumOfSquaresError(ProblemErrorMeasure):
    """
    *Extends:* :class:`ProblemErrorMeasure`

    Calculates a sum-of-squares error: ``f = sum( (x[i] - y[i])**2 )``
    """
    def __call__(self, x):
        return np.sum((self._problem.evaluate(x) - self._values)**2)

