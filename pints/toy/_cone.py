#
# Cone toy log pdf.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
import scipy

from . import ToyLogPDF


class ConeLogPDF(ToyLogPDF):
    r"""
    Toy distribution based on a d-dimensional distribution of the form,

    .. math::

        f(x) \propto e^{-|x|^\beta}

    where ``x`` is a d-dimensional real, and ``|x|`` is the Euclidean norm. The
    mean and variance that are returned relate to expectations on ``|x|`` not
    the multidimensional ``x``.

    Extends :class:`pints.LogPDF`.

    Parameters
    ----------
    dimensions : int
        The dimensionality of the cone.
    beta : float
        The power to which ``|x|`` is raised in the exponential term, which
        must be positive.
    """
    def __init__(self, dimensions=2, beta=1):
        if dimensions < 1:
            raise ValueError('Dimensions must not be less than 1.')
        self._n_parameters = int(dimensions)
        beta = float(beta)
        if beta <= 0:
            raise ValueError('beta must be positive.')
        self._beta = beta

    def beta(self):
        """
        Returns the exponent in the pdf
        """
        return self._beta

    def __call__(self, x):
        if not len(x) == self._n_parameters:
            raise ValueError('x must be of same dimensions as density')
        return -np.linalg.norm(x)**self._beta

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

    def distance(self, samples):
        """
        Calculates a measure of normed distance of samples from exact mean and
        covariance matrix assuming uniform prior with bounds given by
        :meth:`suggested_bounds()`.

        See :meth:`pints.toy.ToyLogPDF.distance()`.
        """
        # Check size of input
        if not len(samples.shape) == 2:
            raise ValueError('Given samples list must be n x 2.')
        if samples.shape[1] != self.n_parameters():
            raise ValueError(
                'Given samples must have length ' +
                str(self.n_parameters()))
        # calculate normed distance
        d = list(map(lambda x: np.linalg.norm(x), samples))
        diff = (
            np.abs(self.mean_normed() - np.mean(d)) +
            np.abs(self.var_normed() - np.var(d))
        )
        return diff

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        L = self.__call__(x)

        norm = np.linalg.norm(x)**2
        norm = self._beta * norm**(-1.0 + self._beta / 2.0)
        dL = np.array([-var * norm for var in x])
        return L, dL

    def mean_normed(self):
        """
        Returns the mean of the normed distance from the origin
        """
        g1 = scipy.special.gamma((1 + self._n_parameters) / self._beta)
        g2 = scipy.special.gamma(self._n_parameters / self._beta)
        return g1 / g2

    def n_parameters(self):
        return self._n_parameters

    def sample(self, n_samples):
        """ See :meth:`ToyLogPDF.sample()`. """
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

    def suggested_bounds(self):
        """ See :meth:`ToyLogPDF.suggested_bounds()`. """
        magnitude = 1000
        bounds = np.tile([-magnitude, magnitude], (self._n_parameters, 1))
        return np.transpose(bounds).tolist()

    def var_normed(self):
        """
        Returns the variance of the normed distance from the origin.
        """
        g1 = scipy.special.gamma((2 + self._n_parameters) / self._beta)
        g2 = scipy.special.gamma(self._n_parameters / self._beta)
        return g1 / g2 - self.mean_normed()**2

