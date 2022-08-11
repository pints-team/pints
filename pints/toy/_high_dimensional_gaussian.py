#
# High-dimensional Gaussian log-pdf.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
import scipy.stats

from . import ToyLogPDF


class HighDimensionalGaussianLogPDF(ToyLogPDF):
    """
    High-dimensional zero-mean multivariate Gaussian log pdf, with off-diagonal
    correlations.

    Specifically, the covariance matrix Sigma is constructed so that diagonal
    elements are integers: Sigma_i,i = i and off-diagonal elements are
    Sigma_i,j = rho * sqrt(i) * sqrt(j).

    Extends :class:`pints.toy.ToyLogPDF`.

    Parameters
    ----------
    dimension : int
        Dimensions of multivariate Gaussian distribution (which must exceed 1).
    rho : float
        The correlation between pairs of parameter dimensions. Note that this
        must be between ```-1 / (dimension - 1) and 1`` so that the
        covariance matrix is positive semi-definite.
    """
    def __init__(self, dimension=20, rho=0.5):
        self._n_parameters = int(dimension)
        if self._n_parameters <= 1:
            raise ValueError('Dimensions must exceed 1.')
        rho = float(rho)
        # bounds must satisfy:
        # https://stats.stackexchange.com/questions/72790/
        if rho > 1.0:
            raise ValueError('rho must be between -1 / (dims - 1) and 1.')
        if rho < -float(1.0 / (self._n_parameters - 1)):
            raise ValueError('rho must be between -1 / (dims - 1) and 1.')
        self._rho = rho

        # Construct mean array
        self._mean = np.zeros(self._n_parameters)

        # Construct covariance matrix
        cov = np.arange(1, 1 + self._n_parameters).reshape(
            (self._n_parameters, 1))
        cov = cov.repeat(self._n_parameters, axis=1)
        cov = np.sqrt(cov)
        cov = self._rho * cov * cov.T
        np.fill_diagonal(cov, 1 + np.arange(self._n_parameters))
        self._cov = cov
        self._cov_inv = np.linalg.inv(cov)

        # Construct scipy 'random variable'
        self._var = scipy.stats.multivariate_normal(self._mean, self._cov)

    def __call__(self, x):
        return self._var.logpdf(x)

    def distance(self, samples):
        """
        Returns approximate Kullback-Leibler divergence between samples
        and underlying distribution.

        See :meth:`pints.toy.ToyLogPDF.distance()`.
        """
        return self.kl_divergence(samples)

    def evaluateS1(self, x):
        """ See :meth:`pints.LogPDF.evaluateS1()`. """
        L = self.__call__(x)
        self._x_minus_mu = x - self._mean

        # derivative wrt x: see https://stats.stackexchange.com/questions/27436/how-to-take-derivative-of-multivariate-normal-density # noqa
        dL = -np.matmul(self._cov_inv, self._x_minus_mu)
        return L, dL

    def kl_divergence(self, samples):
        """
        Returns approximate Kullback-Leibler divergence between samples
        and underlying distribution.

        The returned value is (near) zero for perfect sampling, and then
        increases as the error gets larger.

        See: https://en.wikipedia.org/wiki/Kullback-Leibler_divergence
        """
        # Check size of input
        if not len(samples.shape) == 2:
            raise ValueError('Given samples list must be n x 2.')
        if samples.shape[1] != self._n_parameters:
            raise ValueError(
                'Given samples must have length ' + str(self._n_parameters))

        # Calculate the Kullback-Leibler divergence between the given samples
        # and the multivariate normal distribution underlying this banana.
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
        # For this distribution, s1 is the identify matrix, and m1 is zero,
        # so it simplifies to
        #
        # dkl = 0.5 * (trace(s0) + m0.dot(m0) - log(det(s0)) - k))
        #
        y = np.array(samples, copy=True)
        m0 = np.mean(y, axis=0)
        s0 = np.cov(y.T)
        s1 = self._cov
        m1 = self._mean
        s1_inv = np.linalg.inv(s1)
        return 0.5 * (
            np.trace(np.matmul(s1_inv, s0)) +
            np.matmul(np.matmul(m1 - m0, s1_inv), m1 - m0) -
            np.log(np.linalg.det(s0)) +
            np.log(np.linalg.det(s1)) -
            self._n_parameters)

    def n_parameters(self):
        """ See :meth:`pints.LogPDF.n_parameters()`. """
        return self._n_parameters

    def rho(self):
        """ Returns rho (correlation between dimensions) """
        return self._rho

    def sample(self, n_samples):
        """ See :meth:`pints.toy.ToyLogPDF.sample()`. """
        n_samples = int(n_samples)
        if n_samples < 1:
            raise ValueError(
                'Number of samples must be greater than or equal to 1.')
        return self._var.rvs(n_samples)

    def suggested_bounds(self):
        """ See :meth:`pints.toy.ToyLogPDF.suggested_bounds()`. """
        # maximum variance in one dimension is n_parameters, so use
        # 3 times sqrt of this as prior bounds
        magnitude = 3 * np.sqrt(self.n_parameters())
        bounds = np.tile([-magnitude, magnitude], (self.n_parameters(), 1))
        return np.transpose(bounds).tolist()
