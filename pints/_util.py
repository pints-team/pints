#
# Utility classes for Pints
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from cStringIO import StringIO
import pints
import numpy as np
import sys

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

class _StdOutCapture(object):
    """
    A context manager that redirects and captures the standard output of the
    python interpreter.
    """
    # Based on myokit PyCapture object
    def __init__(self, enabled=True):
        super(_StdOutCapture, self).__init__()
        self._capturing = False     # True if currently capturing
        self._captured = []         # Array to store captured strings in
        self._stdout = None     # Original stdout
        self._dupout = None     # String buffer to redirect stdout to

    def __enter__(self):
        """ Called when the context is entered. """
        self._start_capturing()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Called when exiting the context.
        """
        self._stop_capturing()

    def _start_capturing(self):
        """ Starts capturing output. """
        if not self._capturing:
            # If possible, flush current outputs
            try:
                sys.stdout.flush()
            except AttributeError:
                pass
            # Save current sys stdout
            self._stdout = sys.stdout
            # Create temporary buffers
            self._dupout = StringIO()
            # Re-route
            sys.stdout = self._dupout
            # Now we're capturing!
            self._capturing = True
    
    def _stop_capturing(self):
        """ Stops capturing output. """
        if self._capturing:
            # Flush any remaining output to streams
            self._dupout.flush()
            # Restore original stdout and stderr
            sys.stdout = self._stdout
            # Get captured output
            self._captured.append(self._dupout.getvalue())
            # Delete buffer
            self._dupout = None
            # No longer capturing
            self._capturing = False

    def text(self):
        """
        Disables capturing and returns the captured text.
        """
        self._stop_capturing()
        return ''.join(self._captured)

