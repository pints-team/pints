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
import math

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
    The evaluation of the composed prior assumes the input
    priors are all independent from each other

    For example: `p = ComposedPrior(prior1, prior2, prior2)`.
    """
    def __init__(self, *priors):
        # Check if sub-priors given
        if len(priors) < 1:
            raise ValueError('Must have at least one sub-prior')
        # Check if proper priors, count dimension
        self._dimension = 0
        for prior in priors:
            if not isinstance(prior, Prior):
                raise ValueError('All sub-priors must extend Prior')
            self._dimension += prior.dimension()
        # Store
        self._priors = priors

    def dimension(self):
        return self._dimension

    def __call__(self, x):
        output = 1.0
        lo = hi = 0
        for prior in self._priors:
            lo = hi
            hi += prior.dimension()
            output *= prior(x[lo:hi])
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

class MultivariateNormalPrior(Prior):
    """
    Defines a multivariate normal prior with a given mean and covariance matrix

    For example: `p = NormalPrior(np.array([0,0]),
                                  np.array([[1, 0],[0, 1]]))`
    """
    def __init__(self, mean, cov):
        # Parse input arguments
        if not isinstance(mean, np.array):
            raise ValueError('NormalPrior mean argument requires a numpy array')

        if not isinstance(cov, np.array):
            raise ValueError('NormalPrior cov argument requires a numpy array')

        if mean.ndim != 1:
            raise ValueError('NormalPrior mean must be one dimensional')

        if cov.ndim != 2:
            raise ValueError('NormalPrior cov must be a matrix')

        if mean.shape[0] != cov.shape[0] or mean.shape[0] != cov.shape[1]:
            raise ValueError('mean and cov sizes do not match')

        self._mean = mean
        self._cov = cov
        self._dimension = mean.shape[0]
        self._scipy_normal = scipy.stats.multivariate_normal

    def dimension(self):
        return self._dimension

    def __call__(self, x):
        return self._scipy_normal.pdf(x,mean=self._mean,cov=self._cov)

class NormalPrior(Prior):
    """
    Defines a 1-d normal prior with a given mean and variance

    For example: `p = NormalPrior(0,1)` for a mean of 0 and variance
    of 1
    """
    def __init__(self, mean, cov):
        # Parse input arguments
        self._mean = mean

        # Cache constants
        self._inv2cov = 1.0/(2.0*cov)
        self._scale = 1.0/math.sqrt(2.0*math.pi*cov)

    def dimension(self):
        return 1

    def __call__(self, x):
        return self._scale*math.exp(-self._inv2cov*(x[0]-self._mean)**2)






