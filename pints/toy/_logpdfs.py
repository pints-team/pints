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
