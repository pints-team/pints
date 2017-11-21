#
# Utility classes for Pints
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import numpy as np

def strfloat(x):
    """
    Converts a float to a string, with maximum precision.
    """
    return pints.FLOAT_FORMAT.format(float(x))

def vector(x):
    """
    Copies ``x`` and returns a 1d read-only numpy array of floats with shape
    ``(n,)``.
    Raises a ``ValueError`` if ``x`` has an incompatible shape.
    """
    x = np.array(x, copy=True, dtype=float)
    x.setflags(write=False)
    if x.ndim != 1:
        n = np.max(x.shape)
        if np.prod(x.shape) != n:
            raise ValueError('Unable to convert to 1d vector of scalar values')
        x = x.reshape((n,))
    return x
