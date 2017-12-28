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
import pints
from ._logistic import LogisticModel
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
    def __init__(self, b):
        
        super(TwistedGaussianLogPDF, self).__init__(10)
        self._b = float(b)
        if self._b < 0:
            raise ValueError('b cannot be negative.')

    def __call__(self, x):
        cov = np.diag(np.repeat(1, 10))
        cov[0,0] = 100
        mu = np.array([x[0], x[1] + self._b * x[0]**2 - 100 * self._b,
                       x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]])
        log_pdf = multivariate_normal.logpdf(x, mean=mu, cov=cov)
        return log_pdf



class MultivariateGaussianLogPDF(LogPDF):
    """
    *Extends:* :class:`LogPDF`
    
    Multivariate Gaussian pdf. Default is a covariance
    matrix with cor(i,j) = 0.5, and var(i) = i.
    """
    def __init__(self, dimensions, mu, cov=None):
      
        if dimensions < 1:
            raise ValueError('Dimensions must equal or exceed 1.')
        if len(mu)!= dimensions:
            raise ValueError('Length of mean must equal specified dimensions.')
        
        
        super(MultivariateGaussianLogPDF, self).__init__(dimensions)

    def __call__(self, x):
        cov = np.diag(np.repeat(1, 10))
        cov[0,0] = 100
        mu = np.array([x[0], x[1] + self._b * x[0]**2 - 100 * self._b,
                       x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]])
        log_pdf = multivariate_normal.logpdf(x, mean=mu, cov=cov)
        return log_pdf


class BimodalGaussianLogPDF(LogPDF):
    """
    *Extends:* :class:`LogPDF`
    
    Multivariate Gaussian pdf. Default is a covariance
    matrix with cor(i,j) = 0.5, and var(i) = i.
    """
    def __init__(self, mu1, mu2, cov1=None, cov2=None):
      
        if len(mu1)!= 2:
            raise ValueError('Length of mean must equal 2.')
        if len(mu2)!= 2:
            raise ValueError('Length of mean must equal 2.')

        super(BimodalGaussianLogPDF, self).__init__(2)

    def __call__(self, x):
        cov = np.diag(np.repeat(1, 10))
        cov[0,0] = 100
        mu = np.array([x[0], x[1] + self._b * x[0]**2 - 100 * self._b,
                       x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]])
        log_pdf = multivariate_normal.logpdf(x, mean=mu, cov=cov)
        return log_pdf


class RosenbrockLogPDF(LogPDF):
    """
    *Extends:* :class:`LogPDF`
    
    Rosenbrock function (see: https://en.wikipedia.org/wiki/Rosenbrock_function):
    
    f(x,y) = (a - x)^2 + b(y - x^2)^2
    """
    def __init__(self, a, b):
        self._a = a
        self._b = b
        super(RosenbrockLogPDF, self).__init__(2)

    def __call__(self, x):
        return np.log((self._a - x[0])**2 + self._b * (x[1] - x[0]**2)**2)
