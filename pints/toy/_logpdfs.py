#
# Various toy pdfs to use to test samplers
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
from scipy.stats import multivariate_normal


class LogPDF(object):
    """
    Class for log pdfs. These are typically
    toy problems where the target density
    is known.
    """
    def __init__(self, dimension):
        self._dimension = dimension

    def dimension(self):
        """
        Returns the dimension of the space this pdf is defined on.
        """
        return self._dimension


class TwistedGaussianLogPDF(LogPDF):
    """
    *Extends:* :class:`LogPDF`

    10 dimensional multivariate normal with
    un-normalised density [1]:

    p(x1,x2,x3,...,x10) propto pi(phi(x1,x2,x2,...,x10))

    where pi() is the multivariate normal density and
    phi(x1,x2,x3,...,x10) = (x1,x2+bx1^2-100b,x3,...,x10),
    where b = 0.01/0.1 induces mild/high non-linearity in
    target density.

    [1] "Accelerating Markov Chain Monte Carlo Simulation
    by Differential Evolution with Self-Adaptive Randomized
    Subspace Sampling", 2009, Vrugt et al.,
    International Journal of Nonlinear Sciences and Numerical
    Simulation.
    """
    def __init__(self, b=0.1):

        super(TwistedGaussianLogPDF, self).__init__(10)
        self._b = float(b)
        if self._b < 0:
            raise ValueError('b cannot be negative.')

    def __call__(self, x):
        if len(x) != 10:
            raise ValueError('Dimensions must equal 10')
        cov = np.diag(np.repeat(1, 10))
        cov[0, 0] = 100
        mu = np.repeat(0, 10)
        phi = np.array([x[0], x[1] + self._b * x[0]**2 - 100 * self._b,
                       x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]])
        log_pdf = multivariate_normal.logpdf(phi, mean=mu, cov=cov)
        return log_pdf


class MultivariateGaussianLogPDF(LogPDF):
    """
    *Extends:* :class:`LogPDF`

    Multivariate Gaussian pdf. Default is a covariance
    matrix with cor(i,j) = 0.5, and var(i) = i.
    """
    def __init__(self, dimensions=100, mu=None, cov=None):
        super(MultivariateGaussianLogPDF, self).__init__(dimensions)

        if dimensions < 1:
            raise ValueError('Dimensions must equal or exceed 1.')
        if mu is not None:
            if len(mu) != dimensions:
                raise ValueError('Length of mean must equal ' +
                                 'specified dimensions.')
        if cov is not None:
            dim1, dim2 = cov.shape
            if dim1 != dim2:
                raise ValueError('Covariance matrix must be square')
            if dim1 != dimensions or dim2 != dimensions:
                raise ValueError('Dimensions of covariance matrix ' +
                                 'must match pdf dimensions ' +
                                 '(default is 100).')

        # Construct covariance matrix where diagonal variances = j
        # and off-diagonal covariances = 0.5 * sqrt(i) * sqrt(j)
        if cov is None:
            cor = 0.5 * np.ones((dimensions, dimensions))
            np.fill_diagonal(cor, 1)
            cov = cor
            for i in range(0, dimensions):
                for j in range(0, dimensions):
                    cov[i, j] *= np.sqrt(i + 1) * np.sqrt(j + 1)
        self._cov = cov
        if mu is None:
            mu = np.repeat(0, dimensions)
        self._mu = mu

    def __call__(self, x):
        if len(x) != self._dimension:
            raise ValueError('Dimensions of x must equal pdf dimensions.')
        log_pdf = multivariate_normal.logpdf(x, mean=self._mu, cov=self._cov)
        return log_pdf


class BimodalMultivariateGaussianLogPDF(LogPDF):
    """
    *Extends:* :class:`LogPDF`

    Bimodal multivariate (un-normalised) Gaussian. Default is 2D with modes
    at (0,0) and (10,10) with independent unit
    covariance matrices.
    """
    def __init__(self, dimensions=2, mu1=[0, 0], mu2=[10, 10],
                 cov1=None, cov2=None):
        super(BimodalMultivariateGaussianLogPDF, self).__init__(dimensions)
        if len(mu1) != dimensions:
            raise ValueError('Length of mean must equal ' +
                             'pdf dimensions (default is 2).')
        if len(mu2) != dimensions:
            raise ValueError('Length of mean must equal ' +
                             'pdf dimensions (default is 2).')
        if cov1 is None:
            cov1 = np.diag(np.repeat(1, dimensions))
        else:
            dim1, dim2 = cov1.shape
            if dim1 != dim2:
                raise ValueError('Covariance matrix must be square')
            if dim1 != dimensions or dim2 != dimensions:
                raise ValueError('Dimensions of covariance matrix ' +
                                 'must match pdf dimensions ' +
                                 '(default is 2).')
        if cov2 is None:
            cov1 = np.diag(np.repeat(1, dimensions))
        else:
            dim1, dim2 = cov2.shape
            if dim1 != dim2:
                raise ValueError('Covariance matrix must be square')
            if dim1 != dimensions or dim2 != dimensions:
                raise ValueError('Dimensions of covariance matrix ' +
                                 'must match pdf dimensions ' +
                                 '(default is 2).')
        self._mu1 = mu1
        self._mu2 = mu2
        self._cov1 = cov1
        self._cov2 = cov2

    def __call__(self, x):
        if len(x) != self._dimension:
            raise ValueError('Dimensions of x must equal pdf dimensions.')
        log_pdf = np.log(multivariate_normal.pdf(x,
                                                 mean=self._mu1,
                                                 cov=self._cov1) +
                         multivariate_normal.pdf(x,
                                                 mean=self._mu2,
                                                 cov=self._cov2))
        return log_pdf


class RosenbrockLogPDF(LogPDF):
    """
    *Extends:* :class:`LogPDF`

    Rosenbrock function
    (see: https://en.wikipedia.org/wiki/Rosenbrock_function):

    f(x,y) = -((a - x)^2 + b(y - x^2)^2)

    Note the minus sign converts this from a minimisation to a maximisation
    problem.
    """
    def __init__(self, a=1, b=100):
        self._a = a
        self._b = b
        super(RosenbrockLogPDF, self).__init__(2)

    def __call__(self, x):
        return - np.log((self._a - x[0])**2 + self._b * (x[1] - x[0]**2)**2)
