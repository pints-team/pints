#
# Core modules and methods
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import print_function
from __future__ import division
import pints
import numpy as np

class Boundaries(object):
    """
    Represents a set of lower and upper boundaries for model parameters.
    
    Arguments:
    
    ``lower``
        A 1d array of lower boundaries.
    ``upper``
        The corresponding upper boundaries
    
    """
    def __init__(self, lower, upper):
    
        # Convert to shape (n,) vectors, copy to ensure they remain unchanged
        self._lower = pints.vector(lower)
        self._upper = pints.vector(upper)

        # Get and check dimension    
        self._dimension = len(self._lower)
        if len(self._upper) != self._dimension:
            raise ValueError('Lower and upper bounds must have same length.')

    def check(self, parameters):
        """
        Checks if the given parameter vector is within (or on) the boundaries.
        Raises e
        """
        if np.any(parameters < self._lower):
            return False
        if np.any(parameters > self._upper):
            return False
        return True
    
    def center(self):
        """
        Returns a point in the center of the boundaries.
        """
        return self._lower + 0.5 * (self._upper - self._lower)
    
    def dimension(self):
        """
        Returns the dimension of this set of boundaries.
        """
        return self._dimension
        
    def lower(self):
        """
        Returns the lower boundaries for all parameters (as a read-only NumPy
        array).
        """
        return self._lower

    def upper(self):
        """
        Returns the upper boundary for all parameters (as a read-only NumPy
        array).
        """
        return self._upper
    
    def range(self):
        """
        Returns the size of the parameter space (i.e. `upper - lower`).
        """
        return self._upper - self._lower

class ForwardModel(object):
    """
    Defines an interface for user-supplied forward models.
    
    Classes extending `ForwardModel` can implement the required methods
    directly in Python or interface with other languages (for example via
    Python wrappers around C code).
    """
    
    def __init__(self):
        super(ForwardModel, self).__init__()

    def dimension(self):
        """
        Returns the dimension of the parameter space.
        """
        raise NotImplementedError
        
    def simulate(self, parameters, times):
        """
        Runs a forward simulation with the given ``parameters`` and returns a
        time-series with data points corresponding to the given ``times``.
        
        Arguments:
        
        ``parameters``
            An ordered list of parameter values.
        ``times``
            The times at which to evaluate. Must be an ordered sequence,
            without duplicates, and without negative values.
            All simulations are started at time 0, regardless of whether this
            value appears in ``times``.

        Note: For efficiency, neither ``parameters`` or ``times`` should be
        copied when `simulate` is called.
        """
        raise NotImplementedError

class SingleSeriesProblem(object):
    """
    Represents an inference problem where a model is fit to a single time
    series.
    
    Arguments:
    
    ``model``
        A model or model wrapper extending the `ForwardModel` class.
    ``times``
        A sequence of points in time. See :meth:`model.simulate` for details.
    ``values``
        A sequence of measured (scalar) output values the model should match at
        the given ``times``.
    
    """
    def __init__(self, model, times, values):

        # Check model
        self._model = model
        self._dimension = model.dimension()

        # Check times, copy so that they can no longer be changed and set them
        # to read-only
        self._times = pints.vector(times)
        if np.any(self._times < 0):
            raise ValueError('Times cannot be negative.')
        if np.any(self._times[:-1] > self._times[1:]):
            raise ValueError('Times must be non-decreasing.')

        # Check values, copy so that they can no longer be changed
        self._values = pints.vector(values)
        if len(self._times) != len(self._values):
            raise ValueError('Times and values arrays must have same length.')
    
    def dimension(self):
        """
        Returns the dimensions of this problem.
        """
        return self._dimension
    
    def evaluate(self, parameters):
        """
        Runs a simulation using the given parameters, returning the simulated
        values.
        """
        return self._model.simulate(parameters, self._times)
    
    def times(self):
        """
        Returns this problem's times (as a read-only NumPy array).
        """
        return self._times
    
    def values(self):
        """
        Returns this problem's values (as a read-only NumPy array).
        """
        return self._values
        
def vector(x):
    """
    Copies ``x`` and returns a 1d read-only numpy array of floats with shape
    `(n,)`.
    Raises a `ValueError` if `x` has an incompatible shape.
    """
    x = np.array(x, copy=True, dtype=float)
    x.setflags(write=False)
    if x.ndim != 1:
        n = np.max(x.shape)
        if np.prod(x.shape) != n:
            raise ValueError('Unable to convert to 1d vector of scalar values')
        x = x.reshape((n,))
    return x
