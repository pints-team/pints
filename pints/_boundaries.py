#
# Parameter-space boundaries object
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

