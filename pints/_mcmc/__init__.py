#
# Sub-module containing several MCMC inference routines
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import division
import pints
import numpy as np

class MCMC(object):
    """
    Takes a :class:`LogLikelihood` function and returns a large, representative
    sample.
    
    Arguments:
    
    ``function``
        A :class:`LogLikelihood` function that evaluates points in the
        parameter space.
    ``x0=None``
        An optional starting point for searches in the parameter space.
        
    #TODO
        
    ``sigma0=None``
        An optional initial standard deviation around ``x0``. Can be specified
        either as a scalar value (one standard deviation for all coordinates)
        or as an array with one entry per dimension. Not all methods will use
        this information.
    
    """
    def __init__(self, function, boundaries=None, x0=None, sigma0=None):

        # Store function
        # Likelihood function given? Then wrap an inverter around it
        if isinstance(function, pints.LogLikelihood):
            self._function = pints.LogLikelihoodBasedError(function)
        else:
            self._function = function
        self._dimension = function.dimension()
        
        # Extract bounds
        self._boundaries = boundaries
        if self._boundaries is not None:
            if self._boundaries.dimension() != self._dimension:
                raise ValueError('Boundaries must have same dimension as'
                    ' function.')
        
        # Check initial solution
        if x0 is None:
            # Use value in middle of search space
            if self._boundaries is None:
                self._x0 = np.zeros(self._dimension)
            else:
                self._x0 = 0.5 * (self._boundaries.lower()
                    + self._boundaries.upper())
            self._x0.setflags(write=False)
        else:
            # Check given value
            self._x0 = pints.vector(x0)
            if len(self._x0) != self._dimension:
                raise ValueError('Initial position must have same dimension as'
                    ' function.')
            if self._boundaries is not None:
                if not self._boundaries.check(self._x0):
                    raise ValueError('Initial position must lie within given'
                        ' boundaries.')
        
        # Check initial standard deviation
        if sigma0 is None:
            if self._boundaries:
                # Use boundaries to guess 
                self._sigma0 = (1 / 6.0) * self._boundaries.range()
            else:
                # Use initial position to guess at parameter scaling
                self._sigma0 = (1 / 3.0) * np.abs(self._x0)
                # But add 1 for any initial value that's zero
                self._sigma0 += (self._sigma0 == 0)
            self._sigma0.setflags(write=False)
        
        elif np.isscalar(sigma0):
            # Single number given, convert to vector
            sigma0 = float(sigma0)
            if sigma0 <= 0:
                raise ValueError('Initial standard deviation must be greater'
                    ' than zero.')
            self._sigma0 = np.ones(self._dimension) * sigma0
            self._sigma0.setflags(write=False)
        
        else:
            # Vector given
            self._sigma0 = pints.vector(sigma0)
            if len(self._sigma0) != self._dimension:
                raise ValueError('Initial standard deviation must be None,'
                    ' scalar, or have same dimension as function.')
            if np.any(self._sigma0 <= 0):
                raise ValueError('Initial standard deviations must be greater'
                    ' than zero.')
        
        # Print info to console
        self._verbose = True
        
    def run(self):
        """
        Runs an optimisation and returns the best found value.
        """
        raise NotImplementedError
    
    def set_verbose(self, value):
        """
        Enables or disables verbose mode for this optimiser. In verbose mode,
        lots of output is generated during an optimisation.
        """
        self._verbose = bool(value)
    
    def verbose(self):
        """
        Returns `True` if the optimiser is set to run in verbose mode.
        """
        return self._verbose

