#
# Twisted guassian (banana) distribution toy log pdf.
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


class TwistedGaussianLogPDF(pints.LogPDF):
    """
    *Extends:* :class:`pints.LogPDF`.

    Twisted multivariate normal 'banana' with un-normalised density [1]:

    .. math::
        p(x_1, x_2, x_3, ..., x_n) \propto \pi(\phi(x_1, x_2, x_2, ..., x_n))

    where pi is the multivariate normal density and

    .. math::
        \phi(x_1,x_2,x_3,...,x_n) = (x_1, x_2 + b x_1^2 - V b, x_3, ..., x_n),

    Arguments:

    ``dimension``
        Problem dimension (``n``), must be 2 or greater.
    ``b``
        "Bananicity": ``b = 0.01`` induces mild non-linearity in target
        density, while non-linearity for ``b = 0.1`` is high.
        Must be greater than or equal to zero.
    ``V``
        Offset (see equation).

    [1] Adaptive proposal distribution for random walk Metropolis algorithm
    Haario, Saksman, Tamminen (1999) Computational Statistics.
    """
    def __init__(self, dimension=10, b=0.1, V=100):
        # Check dimension
        self._dimension = int(dimension)
        if self._dimension < 2:
            raise ValueError('Dimension must be 2 or greater.')

        # Check parameters
        self._b = float(b)
        if self._b < 0:
            raise ValueError('Argument `b` cannot be negative.')
        self._V = float(V)

        # Create phi
        self._phi = scipy.stats.multivariate_normal(
            np.zeros(self._dimension), np.eye(self._dimension))

    def __call__(self, x, n_derivatives=0):
        y = np.array(x, copy=True)
        y[1] = x[1] + self._b * ((x[0] ** 2) - self._V)
        y[0] = x[0] / np.sqrt(self._V)
        return self._phi.logpdf(y)

    def n_parameters(self):
        """ See :meth:`pints.LogPDF.n_parameters()`. """
        return self._dimension

