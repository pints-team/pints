#
# German credit toy hierarchical log pdf.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import io
import numpy as np
import scipy
import urllib.request

from . import ToyLogPDF


class GermanCreditHierarchicalLogPDF(ToyLogPDF):
    r"""
    Toy distribution based on a hierarchical logistic regression model, which
    takes the form,

    .. math::

        f(z, y|\beta) \propto \text{exp}(-\sum_{i=1}^{N} \text{log}(1 +
        \text{exp}(-y_i z_i.\beta)) - \beta.\beta/2\sigma^2 -
        N/2 \text{log }\sigma^2 - \lambda \sigma^2)

    The data :math:`(z, y)` are a matrix of individual predictors (with 1s in
    the first column) and responses (1 if the individual should receive credit
    and -1 if not) respectively; :math:`\beta` is a 325x1 vector of
    coefficients and :math:`N=1000`; :math:`z` is the design matrix formed
    by creating all interactions between individual variables and themselves
    as defined in [2]_.

    Extends :class:`pints.LogPDF`.

    Parameters
    ----------
    theta : float
        vector of coefficients of length 326 (first dimension is sigma; other
        entries make up beta)

    References
    ----------
    .. [1] `"UCI machine learning repository"
      <https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)>`_,
      2010. A. Frank and A. Asuncion.
    .. [2] "The No-U-Turn Sampler:  Adaptively Setting Path Lengths in
           Hamiltonian Monte Carlo", 2014, M.D. Hoffman and A. Gelman.
    """
    def __init__(self, x=None, y=None, download=False):
        if x is None or y is None:
            if download is False:
                raise ValueError('No data supplied. Consider setting download'
                                 ' to True to download data.')
            x, y = self._download_data()
            dims = x.shape[1]
        else:
            if download is True:
                raise ValueError(
                    'Either supply no data or set download to True to download'
                    ' data, but not both.')
            dims = x.shape[1]
            if dims != 25:
                raise ValueError('x must have 25 predictor columns.')
            if max(y) != 1 or min(y) != -1:
                raise ValueError('Output must be either 1 or -1.')

        # make design matrix
        self._x = np.copy(x)
        x = x[:, 1:]
        z = np.zeros((1000, 325))
        zz = np.zeros((z.shape[0], 300))
        k = 0
        for i in range(x.shape[1]):
            for j in range(i, x.shape[1]):
                zz[:, k] = np.transpose(x[:, i] * x[:, j])
                k += 1
        zz = np.column_stack([x, zz])
        z[:, 0] = np.ones(1000)
        z[:, 1:] = zz

        self._y = y
        self._z = z
        self._n_parameters = 326
        self._N = len(y)
        self._lambda = 0.01

    def __call__(self, theta):
        sigma = theta[0]
        beta = theta[1::]
        sigma_sq = sigma**2
        log_prob = sum(-np.log(1 + np.exp(-self._y * np.dot(self._z, beta))))
        log_prob += -1 / (2 * sigma_sq) * np.dot(beta, beta) \
                    - self._N / 2 * np.log(sigma) \
                    - self._lambda * sigma_sq
        return log_prob

    def data(self):
        """ Returns data used to fit model: `x`, `y` and `z`."""
        return self._x, self._y, self._z

    def _download_data(self):
        """ Downloads data from [1]. """
        url = ('http://archive.ics.uci.edu/ml/machine-learning-databases/'
               'statlog/german/german.data-numeric')
        url = urllib.request.urlopen(url)
        try:
            raw_data = url.read()
        finally:
            url.close()
        a = np.genfromtxt(io.BytesIO(raw_data), delimiter=4)[:, :25]

        # get output
        y = a[:, -1]
        y[y == 1] = -1
        y[y == 2] = 1

        # get inputs and standardise
        x = a[:, :-1]
        x = scipy.stats.zscore(x)
        x1 = np.zeros((x.shape[0], x.shape[1] + 1))
        x1[:, 0] = np.ones(x.shape[0])
        x1[:, 1:] = x
        x = np.copy(x1)
        return x, y

    def evaluateS1(self, theta):
        """ See :meth:`LogPDF.evaluateS1()`. """
        sigma = theta[0]
        sigma_sq = sigma**2

        beta = theta[1::]
        log_prob = 0.0
        grad_log_prob = np.zeros(self.n_parameters())
        for i in range(self._N):
            exp_yxb = np.exp(-self._y[i] * np.dot(self._z[i], beta))
            log_prob += -np.log(1 + exp_yxb)
            grad_log_prob[1:] += (
                self._z[i] * self._y[i] * exp_yxb / (1 + exp_yxb))

        scale = -1 / (2 * sigma_sq)
        log_prob += scale * np.dot(beta, beta)
        grad_log_prob[1:] += scale * 2 * beta

        log_prob += -self._N / 2 * np.log(sigma_sq)
        grad_log_prob[0] += -self._N / sigma

        log_prob += -self._lambda * sigma_sq
        grad_log_prob[0] += -self._lambda * 2 * sigma

        return log_prob, grad_log_prob

    def n_parameters(self):
        return self._n_parameters

    def suggested_bounds(self):
        """ See :meth:`ToyLogPDF.suggested_bounds()`. """
        magnitude = 100
        bounds = np.tile([-magnitude, magnitude], (self._n_parameters, 1))
        return np.transpose(bounds).tolist()
