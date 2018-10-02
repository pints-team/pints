#
# Cone toy log pdf.
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
import scipy


class ConeLogPDF(pints.LogPDF):
    """
    Toy distribution based on a d-dimensional distribution of the form,

    .. math::

        f(x) \propto e^{-|x|^\\beta}

    where ``x`` is a d-dimensional real, and ``|x|`` is the Euclidean norm. The
    mean and variance that are returned relate to expectations on ``|x|`` not
    the multidimensional ``x``.

    Arguments:

    ``dimensions``
        The dimensionality of the cone.
    ``beta``
        The power to which ``|x|`` is raised in the exponential term, which
        must be positive.

    *Extends:* :class:`pints.LogPDF`.
    """
    def __init__(self, dimensions=2, beta=1):
        if dimensions < 1:
            raise ValueError('Dimensions must not be less than 1.')
        self._n_parameters = int(dimensions)
        beta = float(beta)
        if beta <= 0:
            raise ValueError('beta must be positive.')
        self._beta = beta

    def __call__(self, x):
        return -np.linalg.norm(x)**self._beta

    def n_parameters(self):
        return self._n_parameters

    def beta(self):
        """
        Returns the exponent in the pdf
        """
        return self._beta

    def mean_normed(self):
        """
        Returns the mean of the normed distance from the origin
        """
        g1 = scipy.special.gamma((1 + self._n_parameters) / self._beta)
        g2 = scipy.special.gamma(self._n_parameters / self._beta)
        return g1 / g2

    def var_normed(self):
        """
        Returns the variance of the normed distance from the origin
        """
        g1 = scipy.special.gamma((2 + self._n_parameters) / self._beta)
        g2 = scipy.special.gamma(self._n_parameters / self._beta)
        return g1 / g2 - self.mean_normed()**2

    def CDF(self, x):
        """
        Returns the cumulative density function in terms of ``|x|``.
        """
        x = float(x)
        if x < 0:
            raise ValueError('Normed distance must be non-negative.')
        n = self._n_parameters
        beta = self._beta
        if x == 0:
            return 0
        else:
            return (-(x**n) * ((x**beta)**(-(n / beta))) *
                    scipy.special.gammaincc(n / beta, x**beta) + 1)

    def sample(self, n_samples):
        """
        Generates independent samples from the underlying distribution.
        """
        n_samples = int(n_samples)
        if n_samples < 1:
            raise ValueError(
                'Number of samples must be greater than or equal to 1.')
        n = self._n_parameters

        # Determine empirical inverse-CDF
        x_max = scipy.optimize.minimize(lambda x: (
            np.abs((self.CDF(x) - 1))), 8)['x'][0] * 10
        x_range = np.linspace(0, x_max, 100)
        cdf = [self.CDF(x) for x in x_range]
        f = scipy.interpolate.interp1d(cdf, x_range)

        # Do inverse-transform sampling to obtain independent r samples
        u = np.random.rand(n_samples)
        r = f(u)

        # For each value of r select a value uniformly at random on
        # hypersphere of that radius
        X_norm = np.random.normal(size=(n_samples, n))
        lambda_x = np.sqrt(np.sum(X_norm**2, axis=1))
        x_unit = [r[i] * X_norm[i] / y for i, y in enumerate(lambda_x)]
        return np.array(x_unit)

