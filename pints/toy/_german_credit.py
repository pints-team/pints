#
# German credit toy log pdf.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
from . import ToyLogPDF


class GermanCreditLogPDF(ToyLogPDF):
    r"""
    Toy distribution based on a logistic regression model, which takes the
    form,

    .. math::

        f(x, y|\beta) \propto \text{exp}(-\sum_{i=1}^{N} 1 +
        \text{exp}(-y_i x_i\dot\beta) - 1/2\sigma^2 \beta\dot\beta)

    The data :math:`(x, y)` are a matrix of individual predictors (with 1s in
    the first column) and responses (1 if the individual should receive credit
    and -1 if not) respectively; :math:`\beta` is a 25x1 vector of coefficients
    and :math:`\sigma^2=100`..

    Extends :class:`pints.LogPDF`.

    Parameters
    ----------
    beta : float
        vector of coefficients of length 25.

    References
    ----------
    .. [1] "UCI machine learning repository", 2010. A. Frank and A. Asuncion.
           https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
    """
    def __init__(self, x, y, sigma=10):
        dims = x.shape[1]
        if dims != 25:
            raise ValueError("x must have 25 predictor columns.")
        if max(y) != 1 or min(y) != -1:
            raise ValueError("Output must be either 1 or -1.")
        self._x = x
        self._n_parameters = dims
        self._y = y
        self._sigma = sigma
        self._sigma_sq = sigma**2
        self._N = len(y)

    def __call__(self, beta):
        log_prob = sum(-np.log(1 + np.exp(-self._y * np.dot(self._x, beta))))
        log_prob += -1 / (2 * self._sigma_sq) * np.dot(beta, beta)
        return log_prob

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
