#
# I/O helper classes for Pints
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import sys
try:
    # Python 3
    from io import StringIO
except ImportError:
    # Python2
    from cStringIO import StringIO


class StdOutCapture(object):
    """
    A context manager that redirects and captures the standard output of the
    python interpreter.
    """
    # Based on myokit PyCapture object
    def __init__(self, enabled=True):
        super(StdOutCapture, self).__init__()
        self._capturing = False     # True if currently capturing
        self._captured = []         # Array to store captured strings in
        self._stdout = None     # Original stdout
        self._dupout = None     # String buffer to redirect stdout to

    def __enter__(self):
        """ Called when the context is entered. """
        self._start_capturing()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """ Called when exiting the context. """
        self._stop_capturing()

    def _start_capturing(self):
        """ Starts capturing output. """
        if not self._capturing:
            # If possible, flush current outputs
            try:
                sys.stdout.flush()
            except AttributeError:
                pass
            self._stdout = sys.stdout
            self._dupout = StringIO()
            sys.stdout = self._dupout
            self._capturing = True

    def _stop_capturing(self):
        """ Stops capturing output. """
        if self._capturing:
            self._dupout.flush()
            sys.stdout = self._stdout
            self._captured.append(self._dupout.getvalue())
            self._dupout = None
            self._capturing = False

    def text(self):
        """ Disables capturing and returns the captured text. """
        self._stop_capturing()
        return ''.join(self._captured)
