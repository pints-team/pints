#
# Neal's funnel log pdf.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
import scipy
import scipy.stats

from . import ToyLogPDF


class NealsFunnelLogPDF(ToyLogPDF):
    r"""
    Toy distribution based on a d-dimensional distribution of the form,

    .. math::

        f(x_1, x_2,...,x_d,\nu) =
            \left[\prod_{i=1}^d\mathcal{N}(x_i|0,e^{\nu/2})\right] \times
            \mathcal{N}(\nu|0,3)

    where ``x`` is a d-dimensional real. This distribution was introduced in
    [1]_.

    Extends :class:`pints.toy.ToyLogPDF`.

    Parameters
    ----------
    dimensions : int
        The dimensionality of funnel (by default equal to 10) which must
        exceed 1.

    References
    ----------
    .. [1] "Slice sampling". R. Neal, Annals of statistics, 705 (2003)
           https://doi.org/10.1214/aos/1056562461
    """
    def __init__(self, dimensions=10):
        if dimensions < 2:
            raise ValueError('Dimensions must exceed 1.')
        self._n_parameters = int(dimensions)
        self._s1 = 9.0
        self._s1_inv = 1.0 / self._s1
        self._m1 = 0

    def __call__(self, x):
        if len(x) != self._n_parameters:
            raise ValueError(
                'Length of x must be equal number of parameters')
        nu = x[-1]
        x_temp = x[:-1]
        x_log_pdf = [scipy.stats.norm.logpdf(y, 0, np.exp(nu / 2))
                     for y in x_temp]
        return np.sum(x_log_pdf) + scipy.stats.norm.logpdf(nu, 0, 3)

    def distance(self, samples):
        """ See :meth:`pints.toy.ToyLogPDF.distance()`. """
        return self.kl_divergence(samples)

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        L = self.__call__(x)

        nu = x[-1]
        x_temp = x[:-1]
        cons = np.exp(-nu)
        dnu_first = [0.5 * (cons * var**2 - 1) for var in x_temp]
        dnu = np.sum(dnu_first) - nu / 9.0
        dL = [-var * cons for var in x_temp]
        dL.append(dnu)
        dL = np.array(dL)
        return L, dL

    def kl_divergence(self, samples):
        r"""
        Calculates the KL divergence of samples of the :math:`nu` parameter
        of Neal's funnel from the analytic :math:`\mathcal{N}(0, 3)` result.
        """
        # Check size of input
        if not len(samples.shape) == 2:
            raise ValueError('Given samples list must be n x 2.')
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

    def marginal_log_pdf(self, x, nu):
        r"""
        Yields the marginal density :math:`\text{log } p(x_i,\nu)`.
        """
        return (
            scipy.stats.norm.logpdf(x, 0, np.exp(nu / 2)) +
            scipy.stats.norm.logpdf(nu, 0, 3)
        )

    def mean(self):
        """
        Returns the mean of the target distribution in each dimension.
        """
        return np.zeros(self._n_parameters)

    def n_parameters(self):
        """ See :meth:`pints.LogPDF.n_parameters()`. """
        return self._n_parameters

    def sample(self, n_samples):
        """ See :meth:`pints.toy.ToyLogPDF.sample()`. """
        n = self._n_parameters
        samples = np.zeros((n_samples, n))
        for i in range(n_samples):
            nu = np.random.normal(0, 3, 1)[0]
            sd = np.exp(nu / 2)
            x = np.random.normal(0, sd, n - 1)
            samples[i, 0:(n - 1)] = x
            samples[i, n - 1] = nu
        return samples

    def suggested_bounds(self):
        """ See :meth:`pints.toy.ToyLogPDF.suggested_bounds()`. """
        magnitude = 30
        bounds = np.tile([-magnitude, magnitude],
                         (self._n_parameters, 1))
        return np.transpose(bounds).tolist()

    def var(self):
        """
        Returns the variance of the target distribution in each dimension.
        Note :math:`nu` is the last entry.
        """
        return np.concatenate((np.repeat(90, self._n_parameters - 1), [9]))

