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
    Error measure based on the rosenbrock function (see:
    https://en.wikipedia.org/wiki/Rosenbrock_function):

    .. math::
        f(x,y) = (1 - x)^2 + 100(y - x^2)^2

    *Extends:* :class:`pints.ErrorMeasure`.
    """
    def __init__(self):
        self._a = 1
        self._b = 100

    def n_parameters(self):
        """ See :meth:`pints.ErrorMeasure.n_parameters()`. """
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
    Unnormalised LogPDF based on the Rosenbrock function (see:
    https://en.wikipedia.org/wiki/Rosenbrock_function) although with
    an addition 1 on the denominator to avoid a discontinuity:

    .. math::
        f(x,y) = -log[1 + (1 - x)^2 + 100(y - x^2)^2 ]

    *Extends:* :class:`pints.LogPDF`.
    """
    def __init__(self):
        self._f = RosenbrockError()

        # assumes uniform prior with bounds given by suggested_bounds
        self._true_mean = np.array([0.8693578490590254, 2.599780856590108])
        self._true_cov = np.array([[1.805379677045191, 2.702575590274159],
                                   [2.702575590274159, 8.526583078612177]])

    def n_parameters(self):
        """ See :meth:`pints.LogPDF.n_parameters()`. """
        return self._f.n_parameters()

    def optimum(self):
        """
        Returns the global optimum for this log-pdf.
        """
        return self._f.optimum()

    def __call__(self, x):
        f = (1.0 + self._f(x))
        return -np.log(f)

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`.
        """
        L = self.__call__(x)

        x1 = x[0]
        y1 = x[1]
        a = self._f._a
        b = self._f._b
        dx = -(-2 * (a - x1) - 4 * b * x1 * (y1 - x1**2)) / (
            1 + (a - x1)**2 + b * (y1 - x1**2)**2
        )
        dy = -2 * b * (y1 - x1**2) / (
            1 + (a - x1)**2 + b * (y1 - x1**2)**2
        )

        # derivative wrt x
        dL = np.array([dx, dy])
        return L, dL

    def distance(self, samples):
        """
        Calculates a measure of normed distance of samples from exact mean and
        covariance matrix assuming uniform prior with bounds given
        by `suggested_bounds`
        """
        # Check size of input
        if not len(samples.shape) == 2:
            raise ValueError('Given samples list must be nx2.')
        if samples.shape[1] != self._f.n_parameters():
            raise ValueError(
                'Given samples must have length ' +
                str(self._f.n_parameters()))

        distance = (
            np.linalg.norm(self._true_mean - np.mean(samples, axis=0)) +
            np.linalg.norm(self._true_cov - np.cov(np.transpose(samples)))
        )
        return distance

    def suggested_bounds(self):
        """
        Returns suggested boundaries for prior (typically used in performance
        testing)
        """
        # think the following hard bounds are ok
        bounds = [[-2, 4], [-1, 12]]
        return np.transpose(bounds).tolist()
