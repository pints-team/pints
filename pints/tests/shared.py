#
# Shared classes and methods for testing.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import os
import sys
import shutil
import tempfile
import numpy as np

import pints

# StringIO in Python 3 and 2
try:
    from io import StringIO
except ImportError:
    from cStringIO import StringIO


class StreamCapture(object):
    """
    A context manager that redirects and captures the output stdout, stderr,
    or both.
    """
    def __init__(self, stdout=True, stderr=False):
        super(StreamCapture, self).__init__()

        # True if currently capturing
        self._capturing = False

        # Settings
        self._stdout_enabled = True if stdout else False
        self._stderr_enabled = True if stderr else False

        # Captured output
        self._stdout_captured = None
        self._stderr_captured = None

        # Original streams
        self._stdout_original = None
        self._stderr_original = None

        # Buffers to redirect to
        self._stdout_buffer = None
        self._stderr_buffer = None

    def __enter__(self):
        """ Called when the context is entered. """
        self._start_capturing()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """ Called when exiting the context. """
        self._stop_capturing()

    def _start_capturing(self):
        """ Starts capturing output. """
        if self._capturing:
            return
        self._capturing = True

        # stdout
        if self._stdout_enabled:

            # Create buffer
            self._stdout_buffer = StringIO()

            # Save current stream
            self._stdout_original = sys.stdout

            # If possible, flush current output stream
            try:
                self._stdout_original.flush()
            except AttributeError:
                pass

            # Redirect
            sys.stdout = self._stdout_buffer

        # stderr
        if self._stderr_enabled:

            # Create buffer
            self._stderr_buffer = StringIO()

            # Save current stream
            self._stderr_original = sys.stderr

            # If possible, flush current output stream
            try:
                self._stderr_original.flush()
            except AttributeError:
                pass

            # Redirect
            sys.stderr = self._stderr_buffer

    def _stop_capturing(self):
        """ Stops capturing output. """
        if not self._capturing:
            return

        # stdout
        if self._stdout_enabled:
            self._stdout_buffer.flush()
            sys.stdout = self._stdout_original
            self._stdout_captured = self._stdout_buffer.getvalue()
            self._stdout_buffer = None

        # stderr
        if self._stderr_enabled:
            self._stderr_buffer.flush()
            sys.stderr = self._stderr_original
            self._stderr_captured = self._stderr_buffer.getvalue()
            self._stderr_buffer = None

        self._capturing = False

    def text(self):
        """
        Disables capturing and returns the captured text.

        If only ``stdout`` or ``stderr`` was enabled, a single string is
        returned. If both were enabled a tuple of strings is returned.
        """
        self._stop_capturing()
        if self._stdout_enabled:
            if self._stderr_enabled:
                return self._stdout_captured, self._stderr_captured
            return self._stdout_captured
        return self._stderr_captured    # Could be None


class TemporaryDirectory(object):
    """
    ContextManager that provides a temporary directory to create temporary
    files in. Deletes the directory and its contents when the context is
    exited.
    """
    def __init__(self):
        super(TemporaryDirectory, self).__init__()
        self._dir = None

    def __enter__(self):
        self._dir = os.path.realpath(tempfile.mkdtemp())
        return self

    def path(self, path):
        """
        Returns an absolute path to a file or directory name inside this
        temporary directory, that can be used to write to.

        Example::

            with TemporaryDirectory() as d:
                filename = d.path('test.txt')
                with open(filename, 'w') as f:
                    f.write('Hello')
                with open(filename, 'r') as f:
                    print(f.read())
        """
        if self._dir is None:
            raise RuntimeError(
                'TemporaryDirectory.path() can only be called from inside the'
                ' context.')

        path = os.path.realpath(os.path.join(self._dir, path))
        if path[0:len(self._dir)] != self._dir:
            raise ValueError(
                'Relative path specified to location outside of temporary'
                ' directory.')

        return path

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            shutil.rmtree(self._dir)
        finally:
            self._dir = None

    def __str__(self):
        if self._dir is None:
            return '<TemporaryDirectory, outside of context>'
        else:
            return self._dir


class CircularBoundaries(pints.Boundaries):
    """
    Circular boundaries, to test boundaries that are non-rectangular.

    Arguments:

    ``center``
        The point these boundaries are centered on.
    ``radius``
        The radius (in all directions).

    """
    def __init__(self, center, radius=1):
        super(CircularBoundaries, self).__init__()

        # Check arguments
        center = pints.vector(center)
        if len(center) < 1:
            raise ValueError('Number of parameters must be at least 1.')
        self._center = center
        self._n_parameters = len(center)

        radius = float(radius)
        if radius <= 0:
            raise ValueError('Radius must be greater than zero.')
        self._radius2 = radius**2

    def check(self, parameters):
        """ See :meth:`pints.Boundaries.check()`. """
        return np.sum((parameters - self._center)**2) < self._radius2

    def n_parameters(self):
        """ See :meth:`pints.Boundaries.n_parameters()`. """
        return self._n_parameters

