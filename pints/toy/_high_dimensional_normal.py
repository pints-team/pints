#
# Rosenbrock error measure and log-pdf
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


class HighDimensionalNormalLogPDF(pints.LogPDF):
    """
    *Extends:* :class:`pints.LogPDF`.

    High-dimensional multivariate normal log pdf, with tricky off-diagonal
    covariances.
    """
    def __init__(self, dimension=100):
        self._dimension = int(dimension)
        if self._dimension < 1:
            raise ValueError('Dimension must be 1 or greater.')

        # Construct mean array
        self._mean = np.zeros(self._dimension)

        # Construct covariance matrix where diagonal variances = i
        # and off-diagonal covariances = 0.5 * sqrt(i) * sqrt(j)
        cov = np.arange(1, 1 + self._dimension).reshape((self._dimension, 1))
        cov = cov.repeat(self._dimension, axis=1)
        cov = np.sqrt(cov)
        cov = 0.5 * cov * cov.T
        np.fill_diagonal(cov, 1 + np.arange(self._dimension))
        self._cov = cov

        # Construct scipy 'random variable'
        self._var = scipy.stats.multivariate_normal(self._mean, self._cov)

    def __call__(self, x):
        return self._var.logpdf(x)

    def n_parameters(self):
        """ See :meth:`pints.LogPDF.n_parameters()`. """
        return self._dimension

