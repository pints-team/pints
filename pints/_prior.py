#
# Defines different prior distributions
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import numpy as np

class Prior(object):
    """
    Represents a prior distribution on a vector of variables.
    
    Given any point `x` in the parameter space, a prior function needs to be
    able to evaluate the probability density `f(x)` (such that the integral
    over `f(x)dx` equals 1).
    """
    def dimension(self):
        """
        Returns the dimension of the space this prior is defined on.
        """
        raise NotImplementedError
    def __call__(self, x):
        """
        Returns the probability density for point `x`.
        """
        raise NotImplementedError

class ComposedPrior(Prior):
    """
    Prior composed of one or more sub-priors.
    
    For example: `p = ComposedPrior(prior1, prior2, prior2)`.
    """
    def __init__(self, *priors):
        # Check if sub-priors given
        if len(priors) < 1:
            raise ValueError('Must have at least one sub-prior')
        # Check if proper priors, count dimension
        self._dimensions = []
        for prior in priors:
            if not isinstance(prior, Prior):
                raise ValueError('All sub-priors must extend Prior')
            self._dimensions.append(prior.dimension())
        self._dimension = sum(self._dimensions)
        # Store
        self._priors = priors
    
    def dimension(self):
        return self._dimension
        
    def __call__(self, x):
        output = [0] * self._dimension
        lo = hi = 0
        for i, prior in enumerate(self._priors):
            lo = hi
            hi += self._dimension[i]
            output[lo:hi] = prior(x[lo:hi])
        return output

class UniformPrior(Prior):
    """
    Defines a uniform prior over a given range.
    
    For example: `p = UniformPrior([1,1,1], [10, 10, 100])`, or
    `p = UniformPrior(Boundaries([1,1,1], [10, 10, 100]))`.
    """
    def __init__(self, lower_or_boundaries, upper=None):
        # Parse input arguments
        if upper is None:
            if not isinstance(lower_or_boundaries, pints.Boundaries):
                raise ValueError('UniformPrior requires a lower and an upper'
                    ' bound, or a single Boundaries object.')
            self._boundaries = lower_or_boundaries
        else:
            self._boundaries = pints.Boundaries(lower_or_boundaries, upper)
        # Cache dimension
        self._dimension = self._boundaries.dimension()
        # Cache output value
        self._value = 1.0 / np.product(self._boundaries.range())
    
    def dimension(self):
        return self._dimension
    
    def __call__(self, x):
        return self._value if self._boundaries.check(x) else 0





