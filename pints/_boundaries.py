#
# Parameter-space boundaries object
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class Boundaries(object):
    """
    Represents a set of lower and upper boundaries for model parameters.

    A point ``x`` is considered within the boundaries if (and only if)
    ``lower <= x < upper``.

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

        # Check dimension is at least 1
        if self._dimension < 1:
            raise ValueError('Boundaries must have dimension > 0')

        # Check if upper > lower
        if not np.all(self._upper > self._lower):
            raise ValueError('Upper bounds must exceed lower bounds.')

    def check(self, parameters):
        """
        Checks if the given parameter vector is within the boundaries.
        """
        if np.any(parameters < self._lower):
            return False
        if np.any(parameters >= self._upper):
            return False
        return True

    def n_parameters(self):
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

    def range(self):
        """
        Returns the size of the parameter space (i.e. ``upper - lower``).
        """
        return self._upper - self._lower

    def sample(self, n=1):
        """
        Returns ``n`` random samples from the underlying prior distribution.

        The returned value is a numpy array with shape ``(n, d)`` where ``n``
        is the requested number of samples, and ``d`` is the dimension of the
        parameter space these boundaries are defined on.
        """
        return np.random.uniform(
            self._lower, self._upper, size=(n, self._dimension))

    def upper(self):
        """
        Returns the upper boundary for all parameters (as a read-only NumPy
        array).
        """
        return self._upper

