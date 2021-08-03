#
# Annulus toy log pdf.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
import scipy

from . import ToyLogPDF


class AnnulusLogPDF(ToyLogPDF):
    r"""
    Toy distribution based on a d-dimensional distribution of the form

    .. math::
        f(x|r_0, \sigma) \propto e^{-(|x|-r_0)^2 / {2\sigma^2}}

    where :math:`x` is a d-dimensional real, and :math:`|x|` is the Euclidean
    norm.

    This distribution is roughly a one-dimensional Gaussian distribution
    centred on :math:`r0`, that is smeared over the surface of a hypersphere of
    the same radius. In two dimensions, the density looks like a circular
    annulus.

    Extends :class:`pints.LogPDF`.

    Parameters
    ----------
    dimensions : int
        The dimensionality of the space.
    r0 : float
        The radius of the hypersphere and is approximately the mean normed
        distance from the origin.
    sigma : float
        The width of the annulus; approximately the standard deviation
        of normed distance.
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
        if not len(x) == self._n_parameters:
            raise ValueError('x must be of same dimensions as density')
        return scipy.stats.norm.logpdf(
            np.linalg.norm(x), self._r0, self._sigma)

    def distance(self, samples):
        """
        Calculates a measure of normed distance of samples from exact mean and
        covariance matrix assuming uniform prior with bounds given by
        :meth:`suggested_bounds`.

        See :meth:`ToyLogPDF.distance()`.
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
        dist = (
            np.abs(self.mean_normed() - np.mean(d)) +
            np.abs(self.var_normed() - np.var(d))
        )
        return dist

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`.
        """
        L = self.__call__(x)

        r = self._r0
        norm = np.linalg.norm(x)
        sigma = self._sigma
        cons = -(norm - r) / (norm * sigma**2)
        dL = np.array([var * cons for var in x])
        return L, dL

    def mean(self):
        """
        Returns the mean of this distribution.
        """
        return np.zeros(self._n_parameters)

    def mean_normed(self):
        """
        Returns the mean of the normed distance from the origin.
        """
        return self.moment_normed(1)

    def moment_normed(self, order):
        """
        Returns a given moment of the normed distance from the origin.
        """
        n = self._n_parameters
        r = self._r0
        a = order
        s = self._sigma

        g1 = scipy.special.gamma(0.5 * (n + a))
        g2 = scipy.special.gamma(0.5 * (1 + n + a))
        g3 = scipy.special.gamma(0.5 * n)
        g4 = scipy.special.gamma(0.5 * (1 + n))

        h1 = scipy.special.hyp1f1(0.5 * (n + a), 0.5, r**2 / (2 * s**2))
        h2 = scipy.special.hyp1f1(0.5 * (1 + n + a), 1.5, r**2 / (2 * s**2))
        h3 = scipy.special.hyp1f1(0.5 * (1 - n), 0.5, -r**2 / (2 * s**2))
        h4 = scipy.special.hyp1f1(1 - 0.5 * n, 1.5, -r**2 / (2 * s**2))

        m = 2**(2 - 0.5 * n + 0.5 * (-4 + n + a))
        m *= np.exp(-r**2 / (2 * s**2)) * s**a
        m *= (np.sqrt(2) * s * g1 * h1 + 2 * r * g2 * h2)
        m /= (np.sqrt(2) * s * g3 * h3 + 2 * r * g4 * h4)
        return m

    def n_parameters(self):
        return self._n_parameters

    def r0(self):
        """
        Returns ``r0``.
        """
        return self._r0

    def _reject_sample(self, n_samples):
        """
        Generates non-negative independent samples.
        """
        r = np.ones(n_samples) * -1
        f = r < 0
        while np.any(f):
            r = np.random.normal(self._r0, self._sigma, size=np.sum(f))
            f = r < 0
        return r

    def sample(self, n_samples):
        """ See :meth:`ToyLogPDF.sample()`. """
        n_samples = int(n_samples)
        if n_samples < 1:
            raise ValueError(
                'Number of samples must be greater than or equal to 1.')

        # First sample values of r
        r = self._reject_sample(n_samples)

        # uniformly sample X s.t. their normed distance is r0
        X_norm = np.random.normal(size=(n_samples, self._n_parameters))
        lambda_x = np.sqrt(np.sum(X_norm**2, axis=1))
        x_unit = [r[i] * X_norm[i] / y for i, y in enumerate(lambda_x)]
        return np.array(x_unit)

    def sigma(self):
        """
        Returns ``sigma``
        """
        return self._sigma

    def suggested_bounds(self):
        """ See :meth:`ToyLogPDF.suggested_bounds()`. """
        # in higher dimensions reduce volume as otherwise gets too wide
        r0_magnitude = (self._r0 + self._sigma) * (
            5**(1.0 / (self._n_parameters - 1.0))
        )
        bounds = np.tile([-r0_magnitude, r0_magnitude],
                         (self._n_parameters, 1))
        return np.transpose(bounds).tolist()

    def var_normed(self):
        """
        Returns the variance of the normed distance from the origin.
        """
        return self.moment_normed(2) - self.moment_normed(1)**2

