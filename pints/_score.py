#
# Scoring functions and likelihoods
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import numpy as np

class ErrorMeasure(object):
    """
    Calculates some scalar measure of goodness-of-fit for a model and a data
    set, such that a smaller value means a better fit.
    """
    def __init__(self, problem):
        self._problem = problem
        self._times = problem.times()
        self._values = problem.values()
        self._dimension = problem.dimension()

    def __call__(self, x):
        raise NotImplementedError
        
    def dimension(self):
        """
        Returns the dimension of the space this measure is defined on.
        """
        return self._dimension
    
class SumOfSquaresError(ErrorMeasure):
    """
    Calculates a sum-of-squares error: ``f = sum( (x[i] - y[i])**2 )``
    """        
    def __call__(self, x):
        return np.sum((self._problem.evaluate(x) - self._values)**2)

class RMSError(ErrorMeasure):
    """
    Calculates the square root of a normalised sum-of-squares error:
    ``f = sqrt( sum( (x[i] - y[i])**2 / n) )``
    """
    def __init__(self, problem):
        super(RMSError, self).__init__(self, problem)
        ninv = 1.0 / len(self._values)
    
    def __call__(self, x):
        return np.sqrt(ninv * np.sum(
            (self._problem.evaluate(x) - self._values)**2))

class LogLikelihood(object):
    """
    Represents a log-likelihood that can be used by stochastic methods.
    """
    def __init__(self, problem):
        self._problem = problem
        # Cache some problem variables
        self._values = problem.values()
        self._times = problem.times()
        self._dimension = problem.dimension()

    def __call__(self, x):
        raise NotImplementedError
        
    def dimension(self):
        """
        Returns the dimension of the space this likelihood is defined on.
        """
        return self._dimension

class LogLikelihoodBasedError(ErrorMeasure):
    """
    Inverts a log-likelihood to use it as an error.
    """
    def __init__(self, likelihood):
        if not isinstance(likelihood, LogLikelihood):
            raise ValueError('Argument to LikelihoodBasedError must be'
                ' instance of Likelihood')
    
        super(ErrorMeasure, self).__init__(likelihood._problem)
        self._likelihood = likelihood
    
    def __call__(self, x):
        return -self._likelihood(x)

class GaussianLogLikelihood(LogLikelihood):
    """
    Calculates a log-likelihood based on the assumption of independent
    normally-distributed noise at each time point.
    
    Adds a noise parameter 'sigma' representing the variance of the stochastic
    noise.
    """
    def __init__(self, problem):
        super(GaussianLogLikelihood, self).__init__(problem)
        # Add sneaky parameter to end of list!
        self._dimension = problem.dimension() + 1
        self._size = len(self._times)
        
    def __call__(self, x):
        error = self._values - self._problem.evaluate(x[:-1])
        return -self._size * np.log(x[-1]) - np.sum(error**2) / (2 * x[-1]**2)

    

