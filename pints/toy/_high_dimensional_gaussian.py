#
# High-dimensional Gaussian log-pdf.
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
import scipy.stats


class HighDimensionalGaussianLogPDF(pints.LogPDF):
    """
    High-dimensional multivariate Gaussian log pdf, with tricky off-diagonal
    covariances.

    *Extends:* :class:`pints.LogPDF`.
    """
    def __init__(self, dimension=100):
        self._n_parameters = int(dimension)
        if self._n_parameters < 1:
            raise ValueError('Dimension must be 1 or greater.')

        # Construct mean array
        self._mean = np.zeros(self._n_parameters)

        # Construct covariance matrix where diagonal variances = i
        # and off-diagonal covariances = 0.5 * sqrt(i) * sqrt(j)
        cov = np.arange(1, 1 + self._n_parameters).reshape(
            (self._n_parameters, 1))
        cov = cov.repeat(self._n_parameters, axis=1)
        cov = np.sqrt(cov)
        cov = 0.5 * cov * cov.T
        np.fill_diagonal(cov, 1 + np.arange(self._n_parameters))
        self._cov = cov

        # Construct scipy 'random variable'
        self._var = scipy.stats.multivariate_normal(self._mean, self._cov)

    def __call__(self, x):
        return self._var.logpdf(x)

    def n_parameters(self):
        """ See :meth:`pints.LogPDF.n_parameters()`. """
        return self._n_parameters

