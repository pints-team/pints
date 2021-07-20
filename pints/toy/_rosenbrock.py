#
# Rosenbrock error measure and log-pdf.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
import pints

from . import ToyLogPDF


class RosenbrockError(pints.ErrorMeasure):
    r"""
    Error measure based on the rosenbrock function [1]_.

    .. math::
        f(x,y) = (1 - x)^2 + 100(y - x^2)^2

    Extends :class:`pints.ErrorMeasure`.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Rosenbrock_function
    """
    def __init__(self):
        self._a = 1
        self._b = 100

    def __call__(self, x):
        return (self._a - x[0])**2 + self._b * (x[1] - x[0]**2)**2

    def n_parameters(self):
        """ See :meth:`pints.ErrorMeasure.n_parameters()`. """
        return 2

    def optimum(self):
        """
        Returns the global optimum for this function.
        """
        return self._a, self._a**2


class RosenbrockLogPDF(ToyLogPDF):
    r"""
    Unnormalised LogPDF based on the Rosenbrock function [2]_ with an addition
    of 1 on the denominator to avoid a discontinuity:

    .. math::
        f(x,y) = -log[1 + (1 - x)^2 + 100(y - x^2)^2 ]

    Extends :class:`pints.toy.ToyLogPDF`.

    References
    ----------
    .. [2] https://en.wikipedia.org/wiki/Rosenbrock_function
    """
    def __init__(self):
        self._f = RosenbrockError()

        # assumes uniform prior with bounds given by suggested_bounds
        self._true_mean = np.array([0.8693578490590254, 2.599780856590108])
        self._true_cov = np.array([[1.805379677045191, 2.702575590274159],
                                   [2.702575590274159, 8.526583078612177]])

    def __call__(self, x):
        return -np.log(1.0 + self._f(x))

    def distance(self, samples):
        """
        Calculates a measure of normed distance of samples from exact mean and
        covariance matrix assuming uniform prior with bounds given
        by :meth:`suggested_bounds()`.

        See :meth:`pints.toy.ToyLogPDF.distance()`.
        """
        # Check size of input
        if not len(samples.shape) == 2:
            raise ValueError('Given samples list must be n x 2.')
        if samples.shape[1] != self._f.n_parameters():
            raise ValueError(
                'Given samples must have length ' +
                str(self._f.n_parameters()))

        distance = (
            np.linalg.norm(self._true_mean - np.mean(samples, axis=0)) +
            np.linalg.norm(self._true_cov - np.cov(np.transpose(samples)))
        )
        return distance

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
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

    def n_parameters(self):
        """ See :meth:`pints.LogPDF.n_parameters()`. """
        return self._f.n_parameters()

    def optimum(self):
        """
        Returns the global optimum for this LogPDF.
        """
        return self._f.optimum()

    def suggested_bounds(self):
        """ See :meth:`pints.toy.ToyLogPDF.suggested_bounds()`. """
        # think the following hard bounds are ok
        bounds = [[-2, 4], [-1, 12]]
        return np.transpose(bounds).tolist()
