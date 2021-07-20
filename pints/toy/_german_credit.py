#
# German credit toy log pdf.
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


class GermanCreditLogPDF(ToyLogPDF):
    r"""
    Toy distribution based on a logistic regression model, which takes the
    form,

    .. math::

        f(x, y|\beta) \propto \text{exp}(-\sum_{i=1}^{N} \text{log}(1 +
        \text{exp}(-y_i x_i.\beta)) - \beta.\beta/2\sigma^2)

    The data :math:`(x, y)` are a matrix of individual predictors (with 1s in
    the first column) and responses (1 if the individual should receive credit
    and -1 if not) respectively; :math:`\beta` is a 25x1 vector of coefficients
    and :math:`\sigma^2=100`. The dataset here is from [1]_ but the test
    problem is defined in [2]_.

    Extends :class:`pints.LogPDF`.

    Parameters
    ----------
    beta : float
        vector of coefficients of length 25.

    References
    ----------
    .. [1] `"UCI machine learning repository"
      <https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)>`_,
      2010. A. Frank and A. Asuncion.
    .. [2] "The No-U-Turn Sampler:  Adaptively Setting Path Lengths in
           Hamiltonian Monte Carlo", 2014, M.D. Hoffman and A. Gelman.

    .. _Python: http://www.python.org/
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
        self._x = x
        self._y = y
        self._n_parameters = dims
        self._sigma = 10
        self._sigma_sq = self._sigma**2
        self._N = len(y)

    def __call__(self, beta):
        log_prob = sum(-np.log(1 + np.exp(-self._y * np.dot(self._x, beta))))
        log_prob += -1 / (2 * self._sigma_sq) * np.dot(beta, beta)
        return log_prob

    def data(self):
        """ Returns data used to fit model. """
        return self._x, self._y

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

    def evaluateS1(self, beta):
        """ See :meth:`LogPDF.evaluateS1()`. """
        log_prob = 0.0
        grad_log_prob = np.zeros(self.n_parameters())
        for i in range(self._N):
            exp_yxb = np.exp(-self._y[i] * np.dot(self._x[i], beta))
            log_prob += -np.log(1 + exp_yxb)
            grad_log_prob += self._x[i] * self._y[i] * exp_yxb / (1 + exp_yxb)

        scale = -1 / (2 * self._sigma_sq)
        log_prob += scale * np.dot(beta, beta)
        grad_log_prob += scale * 2 * beta
        return log_prob, grad_log_prob

    def n_parameters(self):
        return self._n_parameters

    def suggested_bounds(self):
        """ See :meth:`ToyLogPDF.suggested_bounds()`. """
        magnitude = 100
        bounds = np.tile([-magnitude, magnitude], (self._n_parameters, 1))
        return np.transpose(bounds).tolist()
