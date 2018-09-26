#
# Annulus toy log pdf.
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


class AnnulusLogPDF(pints.LogPDF):
    """
    Toy distribution based on a d-dimensional distribution of the form,

    .. math::
        f(x|r_0, \sigma) \propto e^{-(|x|-r_0)**2 / {2\sigma**2}}

    where x is a d-dimensional real, and |x| is the Euclidean norm. The mean
    and variance that are returned relate to expectations on |x| not the
    multidimensional x.

    This distribution is roughly a one-dimensional normal distribution centred
    on r0, that is smeared over the surface of a hypersphere of the same
    radius. In two dimensions, the density looks like a circular annulus.

    Arguments:

    ``dimensions``
        The dimensionality of the cone.
    ``r0``
        The radius of the hypersphere and is approximately the mean normed
        distance from the origin.
    ``sigma``
        The width of the annulus; approximately the standard deviation
        of normed distance.

    *Extends:* :class:`pints.LogPDF`.
    """
    def __init__(self, dimensions=2, r0=10, sigma=1):
        if dimensions < 1:
            raise ValueError('Dimensions must not be less than 1.')
        self._n_parameters = int(dimensions)
        r0 = float(r0)
        if r0 <= 0:
            raise ValueError('r0 must be positive.')
        self._r0 = r0
        sigma = float(sigma)
        if sigma <= 0:
            raise ValueError('sigma must be positive.')
        self._sigma = sigma

    def __call__(self, x):
        return scipy.stats.norm.logpdf(np.linalg.norm(x),
                                       self._r0, self._sigma)

    def n_parameters(self):
        return self._n_parameters

    def moment_normed(self, order):
        """
        Returns a given moment of the normed distance from the origin
        """
        n = self._n_parameters
        alpha = order
        r0 = self._r0
        sigma = self._sigma
        front = 2**(2 - 0.5 * n + 0.5 * (-4 + n + alpha)) * (
            np.exp(-r0**2 / (2 * sigma**2)) * sigma**(alpha))
        first_parenthesis = (np.sqrt(2) * sigma *
                             scipy.special.gamma(0.5 * (n + alpha)) *
                             scipy.special.hyp1f1(0.5 * (n + alpha),
                                                  0.5,
                                                  r0**2 / (2 * sigma**2)) +
                             2 * r0 *
                             scipy.special.gamma(0.5 * (1 + n + alpha)) *
                             scipy.special.hyp1f1(0.5 * (1 + n + alpha),
                                                  1.5, r0**2 / (2 * sigma**2)))
        denominator = (np.sqrt(2) * sigma * scipy.special.gamma(0.5 * n) *
                       scipy.special.hyp1f1(0.5 * (1 - n),
                                            0.5, -r0**2 / (2 * sigma**2)) +
                       2 * r0 * scipy.special.gamma(0.5 * (1 + n)) *
                       scipy.special.hyp1f1(1 - 0.5 * n,
                       1.5, -r0**2 / (2 * sigma**2)))
        return front * first_parenthesis / denominator

    def mean_normed(self):
        """
        Returns the mean of the normed distance from the origin
        """
        return self.moment_normed(1)

    def var_normed(self):
        """
        Returns the variance of the normed distance from the origin
        """
        a_mean = self.moment_normed(1)
        return self.moment_normed(2) - a_mean**2

    def sample(self, n_samples):
        """
        Generates independent samples from the underlying distribution.
        """
        n_samples = int(n_samples)
        if n_samples < 1:
            raise ValueError('Number of samples must be greater than or ' +
                             'equal to 1.')
        # first sample values of r
        r = [self._reject_sample_r() for i in range(n_samples)]
        r = np.array(r)
        # uniformly sample X s.t. their normed distance is r0
        X_norm = np.random.normal(size=(n_samples, self._n_parameters))
        lambda_x = np.sqrt(np.sum(X_norm**2, axis=1))
        x_unit = [r[i] * X_norm[i] / y for i, y in enumerate(lambda_x)]
        return np.array(x_unit)

    def _reject_sample_r(self):
        """
        Generates a non-negative independent sample of r
        """
        r = np.random.normal(loc=self._r0, scale=self._sigma, size=1)
        while r < 0:
            r = np.random.normal(loc=self._r0, scale=self._sigma, size=1)
        return r
