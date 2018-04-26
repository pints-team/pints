#
# I/O helper classes for Pints
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
import pints
import numpy as np
try:
    # Python 3
    from io import StringIO
except ImportError:
    # Python2
    import cStringIO.StringIO as StringIO


def load_samples(filename, n=None):
    """
    Loads samples from the given ``filename`` and returns a 2d numpy array
    containing them.

    If the optional argument ``n`` is given, the method assumes there are ``n``
    files, with names based on ``filename`` such that e.g. ``test.csv`` would
    become ``test_0.csv``, ``test_1.csv``, ..., ``test_n.csv``. In this case
    a list of 2d numpy arrays is returned.

    Assumes the first line in each file is a header.

    See also :meth:`save_samples()`.
    """
    # Define data loading method
    def load(filename):
        with open(filename, 'r') as f:
            lines = iter(f)
            next(lines)  # Skip header
            return np.asarray(
                [[float(x) for x in line.split(',')] for line in lines])

    # Load from filename directly
    if n is None:
        return load(filename)

    # Load from systematically named files
    n = int(n)
    if n < 1:
        raise ValueError(
            'Argument `n` must be `None` or an integer greater than zero.')
    parts = os.path.splitext(filename)
    filenames = [parts[0] + '_' + str(i) + parts[1] for i in range(n)]

    # Check if files exist before loading (saves times)
    for filename in filenames:
        if not os.path.isfile(filename):
            try:
                # Python 3
                raise FileNotFoundError('File not found: ' + filename)
            except NameError:
                # Python 2
                raise IOError('File not found: ' + filename)

    # Load and return
    return [load(filename) for filename in filenames]


def save_samples(filename, *sample_lists):
    """
    Stores one or multiple lists of samples at the path given by ``filename``.

    If one list of samples is given, the filename is used as is. If multiple
    lists are given, the filenames are updated to include ``_0``, ``_1``,
    ``_2``, etc.

    For example, ``save_samples('test.csv', samples)`` will store information
    from ``samples`` in ``test.csv``. Using
    ``save_samples('test.csv', samples_0, samples_1)`` will store the samples
    from ``samples_0`` to ``test_0.csv`` and ``samples_1`` to ``test_1.csv``.

    See also: :meth:`load_samples()`.
    """
    # Get filenames
    k = len(sample_lists)
    if k < 1:
        raise ValueError('At least one set of samples must be given.')
    elif k == 1:
        filenames = [filename]
    else:
        parts = os.path.splitext(filename)
        filenames = [parts[0] + '_' + str(i) + parts[1] for i in range(k)]

    # Check shapes
    i = iter(sample_lists)
    shape = np.asarray(next(i)).shape
    if len(shape) != 2:
        raise ValueError(
            'Samples must be given as 2d arrays (e.g. lists of lists).')
    for samples in i:
        if np.asarray(samples).shape != shape:
            raise ValueError('All sample lists must have same shape.')

    # Store
    filename = iter(filenames)
    header = ','.join(['"p' + str(j) + '"' for j in range(shape[1])])
    for samples in sample_lists:
        with open(next(filename), 'w') as f:
            f.write(header + '\n')
            for sample in samples:
                f.write(','.join([pints.strfloat(x) for x in sample]) + '\n')


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
        self._dir = tempfile.mkdtemp()
        return self

    def path(self, path):
        """
        Returns an absolute path to a file or directory name inside this
        temporary directory, that can be used to write to.

        Example::

            with pints.io.TemporaryDirectory() as d:
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

