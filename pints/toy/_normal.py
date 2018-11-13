#
# Unimodal Normal/Gaussian toy log pdf.
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
import scipy.stats


class NormalLogPDF(pints.LogPDF):
    """
    Toy distribution based on a multivariate (unimodal) Normal/Gaussian
    distribution.

    Arguments:

    ``mean``
        The distribution mean (specified as a vector).
    ``sigma``
        The distribution's covariance matrix. Can be given as either a matrix
        or a vector (in which case ``diag(sigma)`` will be used. Should be
        symmetric and positive-semidefinite.

    *Extends:* :class:`pints.LogPDF`.
    """
    def __init__(self, mean, sigma):

        # Copy and convert
        mean = np.array(mean, copy=True)
        sigma = np.array(sigma, copy=True)

        # Check dimension
        self._n_parameters = len(mean)
        if sigma.shape == (self._n_parameters, ):
            sigma = np.diag(sigma)
        elif sigma.shape != (self._n_parameters, self._n_parameters):
            raise ValueError(
                'Sigma must have same dimension as mean, or be a square matrix'
                ' with the same dimension as the mean.')

        # Store
        self._mean = mean
        self._sigma = sigma

        # Create scipy distribution
        self._phi = scipy.stats.multivariate_normal(self._mean, self._sigma)

    def __call__(self, x):
        return self._phi.logpdf(x)

    def kl_divergence(self, samples):
        """
        Calculates the Kullback-Leibler divergence between a given list of
        samples and the distribution underlying this LogPDF.

        The returned value is (near) zero for perfect sampling, and then
        increases as the error gets larger.

        See: https://en.wikipedia.org/wiki/Kullback-Leibler_divergence
        """
        # Check size of input
        if not len(samples.shape) == 2:
            raise ValueError('Given samples list must be 2x2.')
        if samples.shape[1] != self._n_parameters:
            raise ValueError(
                'Given samples must have length ' + str(self._n_parameters))

        # Calculate the Kullback-Leibler divergence between the given samples
        # and this multivariate normal distribution underlying.
        # From wikipedia:
        #
        # k = dimension of distribution
        # dkl = 0.5 * (
        #       trace(s1^-1 * s0)
        #       + (m1 - m0)T * s1^-1 * (m1 - m0)
        #       + log( det(s1) / det(s0) )
        #       - k
        #       )
        #
        # using s1 = real sigma, as this needs to be inverted and the real one
        # is more likely to be invertible than the sample one
        m0 = np.mean(samples, axis=0)
        m1 = self._mean
        s0 = np.cov(samples.T)
        s1 = self._sigma
        cov_inv = np.linalg.inv(s1)

        dkl1 = np.trace(cov_inv.dot(s0))
        dkl2 = np.dot((m1 - m0).T, cov_inv).dot(m1 - m0)
        dkl3 = np.log(np.linalg.det(s1) / np.linalg.det(s0))
        return 0.5 * (dkl1 + dkl2 + dkl3 - self._n_parameters)

    def n_parameters(self):
        """ See :meth:`pints.LogPDF.n_parameters()`. """
        return self._n_parameters

    def sample(self, n):
        """
        Generates samples from the underlying distribution.
        """
        if n < 0:
            raise ValueError('Number of samples cannot be negative.')
        return self._phi.rvs(n)

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`.
        """
        L = self.__call__(x)

        self._sigma_inv = np.linalg.inv(self._sigma)
        self._x_minus_mu = x - self._mean

        # derivative wrt x
        dL = -np.matmul(self._sigma_inv, self._x_minus_mu)
        return L, dL

