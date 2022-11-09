#
# Shared classes and methods for testing.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import io
import os
import shutil
import sys
import tempfile

import numpy as np

import pints


class StreamCapture(object):
    """
    A context manager that redirects and captures the output stdout, stderr,
    or both.

    Warning: This class is not thread-safe.
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
            self._stdout_buffer = io.StringIO()

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
            self._stderr_buffer = io.StringIO()

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


class SubCapture(object):
    """
    A context manager that redirects and captures the standard and error output
    of the current process, using low-level file descriptor duplication.

    This can be useful to capture output that comes from C extensions, e.g. in
    the interface classes.

    The argument ``dump_on_error`` can be set to ``True`` to print all output
    if an error occurs while the context manager is active.

    Warning: This class is not thread-safe.
    """
    def __init__(self, dump_on_error=False):
        super(SubCapture, self).__init__()
        self._capturing = False
        self._captured = []
        self._dump_on_error = bool(dump_on_error)
        self._stdout = None     # Original stdout object
        self._stderr = None     # Original stderr object
        self._stdout_fd = None  # Original file descriptor used for output
        self._stderr_fd = None  # Original file descriptor used for errors
        self._dupout_fd = None  # Back-up of file descriptor for output
        self._duperr_fd = None  # Back-up of file descriptor for errors
        self._file_out = None   # Temporary file to write output to
        self._file_err = None   # Temporary file to write errors to

    def __enter__(self):
        """ Called when entering the context. """
        self._start_capturing()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """ Called when exiting the context. """
        self._stop_capturing()

        # Error? Then print all output
        if (self._dump_on_error and exc_type is not None):
            print(self.text())

    def _start_capturing(self):
        """
        Starts capturing output to stdout and stderr.
        """
        if not self._capturing:
            # If possible, flush original outputs
            try:
                sys.stdout.flush()
            except AttributeError:  # pragma: no cover
                pass
            try:
                sys.stderr.flush()
            except AttributeError:  # pragma: no cover
                pass

            # Save any redirected output / error streams
            self._stdout = sys.stdout
            self._stderr = sys.stderr

            # Get file descriptors used for output and errors.
            #
            # On https://docs.python.org/3/library/sys.html#module-sys, it says
            # that stdout/err as well as __stdout__ can be None (e.g. in spyder
            # on windows), so we need to check for this.
            # In other cases (pythonw.exe) they can be set but return a
            # negative file descriptor (indicating it's invalid).
            # So here we check if __stdout__ is None and if so set a negative
            # fileno so that we can catch both cases at once in the rest of the
            # code.
            #
            if sys.__stdout__ is not None:
                self._stdout_fd = sys.__stdout__.fileno()
            else:   # pragma: no cover
                self._stdout_fd = -1
            if sys.__stderr__ is not None:
                self._stderr_fd = sys.__stderr__.fileno()
            else:   # pragma: no cover
                self._stderr_fd = -1

            # If they're proper streams (so if not pythonw.exe), flush them
            if self._stdout_fd >= 0:
                sys.stdout.flush()
            if self._stderr_fd >= 0:
                sys.stderr.flush()

            # Create temporary files
            # Make sure this isn't opened in binary mode, and specify +
            # for reading and writing.
            self._file_out = tempfile.TemporaryFile(mode='w+')
            self._file_err = tempfile.TemporaryFile(mode='w+')

            # Redirect python-level output to temporary files
            # (Doing this is required to make this work on windows)
            sys.stdout = self._file_out
            sys.stderr = self._file_err

            # If possible, pipe the original output and errors to files
            # On windows, the order is important: First dup both stdout and
            # stderr, then dup2 the new descriptors in. This prevents a weird
            # infinite recursion on windows ipython / python shell.
            self._dupout_fd = None
            self._duperr_fd = None
            if self._stdout_fd >= 0:
                self._dupout_fd = os.dup(self._stdout_fd)
            if self._stderr_fd >= 0:
                self._duperr_fd = os.dup(self._stderr_fd)
            if self._stdout_fd >= 0:
                os.dup2(self._file_out.fileno(), self._stdout_fd)
            if self._stderr_fd >= 0:
                os.dup2(self._file_err.fileno(), self._stderr_fd)

            # Now we're capturing!
            self._capturing = True

    def _stop_capturing(self):
        """
        Stops capturing output. If capturing was already halted, this does
        nothing.
        """
        if self._capturing:
            # Flush any remaining output
            sys.stdout.flush()
            sys.stderr.flush()
            # Undo dupes, if made
            if self._dupout_fd is not None:
                os.dup2(self._dupout_fd, self._stdout_fd)
                os.close(self._dupout_fd)
            if self._duperr_fd is not None:
                os.dup2(self._duperr_fd, self._stderr_fd)
                os.close(self._duperr_fd)
            # Reset python-level redirects
            sys.stdout = self._stdout
            sys.stderr = self._stderr
            # Close temporary files and store capture output
            try:
                self._file_out.seek(0)
                self._captured.extend(self._file_out.readlines())
                self._file_out.close()
            except ValueError:  # pragma: no cover
                # In rare cases, I've seen a ValueError, "underlying buffer has
                # been detached".
                pass
            try:
                self._file_err.seek(0)
                self._captured.extend(self._file_err.readlines())
                self._file_err.close()
            except ValueError:  # pragma: no cover
                pass
            # We've stopped capturing
            self._capturing = False

    def text(self):
        return ''.join(self._captured)


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

    Parameters
    ----------
    center
        The point these boundaries are centered on.
    radius
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


class SwappingTransformation(pints.Transformation):
    """
    Transformation that swaps the parameters around (as a very simple example
    of a transformation that isn't elementwise).
    """
    def __init__(self, n_parameters):
        self._n_parameters = int(n_parameters)

    def elementwise(self):
        return False

    def n_parameters(self):
        return self._n_parameters

    def to_model(self, q):
        return q[::-1]

    def to_search(self, p):
        return p[::-1]
