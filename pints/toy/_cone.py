#
# Unimodal Normal/Gaussian toy log pdf.
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
import scipy


class ConeLogPDF(pints.LogPDF):
    """
    Toy distribution based on a d-dimensional distribution of the form,

    .. math::
        f(x) \propto e^{-|x|^\beta}

    where x is a d-dimensional real, and |x| is the Euclidean norm. The mean
    and variance that are returned relate to expectations on |x| not the
    multidimensional x.

    Arguments:

    ``dimensions``
        The dimensionality of the cone.
    ``beta``
        The power to which |x| is raised in the exponential term, which must be
        positive.

    *Extends:* :class:`pints.LogPDF`.
    """
    def __init__(self, dimensions=2, beta=1):
        if dimensions < 1:
            raise ValueError('Dimensions must not be less than 1.')
        if not isinstance(dimensions, int):
            raise ValueError('Dimensions must be integer.')
        self._n_parameters = dimensions
        if beta <= 0:
            raise ValueError('beta must be positive.')
        self._beta = beta

    def __call__(self, x):
        return -np.linalg.norm(x)**self._beta

    def n_parameters(self):
        return self._n_parameters

    def beta(self):
        """
        Returns the exponent in the pdf
        """
        return self._beta

    def mean_normed(self):
        """
        Returns the mean of the normed distance from the origin
        """
        return (scipy.special.gamma((1 + self._n_parameters) / self._beta) /
                scipy.special.gamma(self._n_parameters / self._beta))

    def var_normed(self):
        """
        Returns the variance of the normed distance from the origin
        """
        return ((scipy.special.gamma((2 + self._n_parameters) / self._beta) /
                scipy.special.gamma(self._n_parameters / self._beta)) -
                self.mean_normed()**2)
