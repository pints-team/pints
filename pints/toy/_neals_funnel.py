#
# Neal's funnel log pdf.
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
import scipy
import scipy.stats


class NealsFunnelLogPDF(pints.LogPDF):
    """
    Toy distribution based on a d-dimensional distribution of the form,

    .. math::

        f(x_1, x_2,...,x_d,\\nu) =
            \\left[\\prod_{i=1}^d\\mathcal{N}(x_i|0,e^{\\nu/2})\\right] \\times
            \\mathcal{N}(\\nu|0,3)

    where ``x`` is a d-dimensional real.

    Arguments:

    ``dimensions``
        The dimensionality of funnel (by default equal to 10) which must
        exceed 1.

    *Extends:* :class:`pints.LogPDF`.
    """
    def __init__(self, dimensions=10):
        if dimensions < 2:
            raise ValueError('Dimensions must exceed 1.')
        self._n_parameters = int(dimensions)
        self._s1 = 9.0
        self._s1_inv = 1.0 / self._s1
        self._m1 = 0

    def __call__(self, x):
        nu = x[-1]
        x_temp = x[:-1]
        x_log_pdf = [scipy.stats.norm.logpdf(y, 0, np.exp(nu / 2))
                     for y in x_temp]
        return np.sum(x_log_pdf) + scipy.stats.norm.logpdf(nu, 0, 3)

    def n_parameters(self):
        """ See :meth:`pints.LogPDF.n_parameters()`. """
        return self._n_parameters

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`.
        """
        L = self.__call__(x)

        nu = x[-1]
        x_temp = x[:-1]
        dnu = -nu / 9.0
        cons = -np.exp(-nu)
        dL = [var * cons for var in x_temp]
        dL.append(dnu)
        return L, dL

    def kl_divergence(self, samples):
        """
        Calculates the KL divergence of samples of the :math:`nu` parameter
        of Neal's funnel from the analytic :math:`\\mathcal{N}(0, 3)` result
        """
        # Check size of input
        if not len(samples.shape) == 2:
            raise ValueError('Given samples list must be nx2.')
        if samples.shape[1] != self._n_parameters:
            raise ValueError(
                'Given samples must have length ' + str(self._n_parameters))
        nu = samples[:, self._n_parameters - 1]
        m0 = np.mean(nu)
        s0 = np.var(nu)

        return 0.5 * (np.sum(self._s1_inv * s0) +
                      (self._m1 - m0) * self._s1_inv * (self._m1 - m0) -
                      np.log(s0) +
                      np.log(self._s1) -
                      1)

    def mean(self):
        """
        Returns the mean of the target distribution in each dimension.
        """
        return np.zeros(self._n_parameters)

    def var(self):
        """
        Returns the variance of the target distribution in each dimension.
        Note :math:`nu` is the last entry.
        """
        return np.concatenate((np.repeat(90, self._n_parameters), [9]))

    def sample(self, n_samples):
        """ Samples from the underlying distribution. """
        n = self._n_parameters
        samples = np.zeros((n_samples, n))
        for i in range(n_samples):
            nu = np.random.normal(0, 3, 1)[0]
            sd = np.exp(nu / 2)
            x = np.random.normal(0, sd, n - 1)
            samples[i, 0:(n - 1)] = x
            samples[i, n - 1] = nu
        return samples
