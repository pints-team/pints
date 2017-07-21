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

class MeasureOfFit():
    """
    Calculates some scalar measure of goodness-of-fit for a model and a data
    set, for example a scoring function or a log-likelihood.
    
    More positive values mean worse fits.
    """
    def __init__(self, problem):
        self._problem = problem
        self._dimension = problem._dimension
        
    def __call__(self, x):
        raise NotImplementedError
    
class SumOfSquaresError(MeasureOfFit):
    """
    Calculates a sum-of-squares error: ``f = sum( (x[i] - y[i])**2 )``
    """        
    def __call__(self, x):
        return np.sum((self._problem.evaluate(x) - self._problem._values)**2)

class RMSError(MeasureOfFit):
    """
    Calculates the square root of a normalised sum-of-squares error:
    ``f = sqrt( sum( (x[i] - y[i])**2 / n) )``
    """
    def __init__(self, problem):
        super(RMSError, self).__init__(self, problem)
        ninv = 1.0 / len(self._problem._values)
    
    def __call__(self, x):
        return np.sqrt(ninv * np.sum(
            (self._problem.evaluate(x) - self._problem._values)**2))
        

