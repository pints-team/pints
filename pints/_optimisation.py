#
# Shared classes and methods for optimisers
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import numpy as np

class Optimiser(object):
    """
    Takes a model and recorded data as input and attempts to find the model
    parameters that best reproduce the recordings.
    
    Arguments:
    
    ``function``
        A :class:`MeasureOfFit` function that evaluates points in the parameter
        space.
    ``boundaries=None``
        An optional set of boundaries on the parameter space.
    ``hint=None``
        An optional starting point for searches in the parameter space.
    
    """
    def __init__(self, function, boundaries=None, hint=None):
        
        # Store function
        self._function = function
        self._dimension = function._dimension
        
        # Extract bounds
        self._boundaries = boundaries
        if self._boundaries is not None:
            if self._boundaries._dimension != self._dimension:
                raise ValueError('Boundaries must have same dimension as'
                    ' function.')
        
        # Check hint
        if hint is None:
            # Use value in middle of search space
            if self._boundaries is None:
                self._hint = np.zeros(self._dimension)
            else:
                self._hint = 0.5 * (self._boundaries._lower
                    + self._boundaries._upper)
        else:
            # Check given value
            self._hint = pints.vector(hint)
            if len(self._hint) != self._dimension:
                raise ValueError('Hint must have same dimension as'
                    ' function.')
            if self._boundaries is not None:
                if not self._boundaries.check(hint):
                    raise ValueError('Hint must lie within given boundaries.')
        
    def run(self):
        """
        Runs an optimisation and returns the best found value.
        """
        raise NotImplementedError

class TriangleWaveTransform(object):
    """
    Transforms from unbounded to bounded parameter space using a periodic
    triangle-wave transform.
    
    Can be applied to single values or arrays of values.
    """
    def __init__(self, boundaries):
        self._lower = boundaries._lower
        self._upper = boundaries._upper
        self._range = self._upper - self._lower
        self._range2 = 2 * self._range

    def __call__(self, x, *args):
        y = np.remainder(x - self._lower, self._range2)
        z = np.remainder(y, self._range)
        return ((self._lower + z) * (y < self._range)
            + (self._upper - z) * (y >= self._range))



