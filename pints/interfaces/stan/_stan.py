#
# Interface for Stan models
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
from collections import Counter
import pystan
import pints
import warnings


import sys
import os
import tempfile
class SubCapture(object):
    """
    A context manager that redirects and captures the standard and error output
    of the current process, using low-level file descriptor duplication.

    This can be useful to capture output that comes from C extensions, e.g. in
    the interface classes.

    The argument ``dump_on_error`` can be set to ``True`` to print all output
    if an error occurs while the context manager is active.
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


SLOW = True


class StanLogPDF(pints.LogPDF):
    def __init__(self, stan_code, stan_data=None):
        """
        Creates a :class:`pints.LogPDF` object from Stan code and data, which
        can then be used in sampling, optimisation etc.

        Note, that this class uses Pystan to interface with Stan which then
        goes on to compile the underlying Stan model (see [1]_), so creating a
        `StanLogPDF` can take some time (typically minutes or so).

        Note that the interface assumes that the parameters are on the
        unconstrained scale (according to Stan's "constraint transforms" [1]_).
        So, for example, if a variable is declared to have a lower bound of
        zero, sampling happens on the log-transformed space. The interface
        takes care of Jacobian transformations, so a user only needs to
        transform the variable back to the constrained space (in the example,
        using an ``exp`` transform) to obtain appropriate samples.

        Extends :class:`pints.LogPDF`.

        Parameters
        ----------
        stan_code
            Stan code describing the model.
        stan_data
            Data in Python dictionary format as required by PyStan. Defaults to
            None in which case ``update_data`` must be called to create a valid
            Stan model fit object before calling.

        References
        ----------
        .. [1] "Stan: a probabilistic programming language".
               B Carpenter et al., (2017), Journal of Statistical Software
        """
        self._fit = None
        self._log_prob = None
        self._grad_log_prob = None
        self._n_parameters = None
        self._names = None
        self._index = None
        self._long_names = None
        self._counter = None
        self._dict = None

        # Compile stan model
        if SLOW:
            with SubCapture():
                self._compiled_stan = pystan.StanModel(model_code=stan_code)

        # Create stanfit if data is supplied
        if stan_data is not None:
            self.update_data(stan_data)

    def names(self):
        """ Returns names of Stan parameters. """
        return self._long_names

    def n_parameters(self):
        """ See `pints.LogPDF.n_parameters`. """
        return self._n_parameters

    def update_data(self, stan_data):
        """
        Updates data passed to the underlying Stan model.

        Parameters
        ----------
        stan_data
            Data in Python dictionary format as required by PyStan.
        """
        if SLOW:
            with SubCapture():
                self._fit = self._compiled_stan.sampling(
                    data=stan_data,
                    iter=1,
                    chains=1,
                    verbose=False,
                    refresh=0,
                    control={'adapt_engaged': False}
                )

        self._log_prob = self._fit.log_prob
        self._grad_log_prob = self._fit.grad_log_prob

        # Get parameter names, in the order that they are specified in the
        # model. Vector parameters will appear multiple times, with a '.i'
        # suffix indicating the indice for each.
        self._long_names = self._fit.unconstrained_param_names()

        print('Initialise')
        print('long names', self._long_names)

        # Number of parameters equals number of parameter names (so vectorised
        # ones are potentially counted more than once)
        self._n_parameters = len(self._long_names)

        # Create short name list and mapping
        self._names, self._index = self._initialise_dict_index(self._long_names)
        # At this point, _names is a randomly sorted list of unique suffix-free
        # names (of length <= n_parameters), and while _index is a list of
        # length n_parameters, where the i-th index corresponds to the i-th
        # index in the parameter vector and in _long_names, while its value is
        # the index of the corresponding short name

        print('self._names', self._names)
        print('self._index', self._index)


        self._counter = Counter(self._index)
        # After this, _counter will be a dict mapping each index (ranging from
        # 0 to len(_names)-1), to the number of times it appears in the longer
        # parameter vector. The sum of all entries will equal n_parameters

        self._dict = {self._names[i]: [] for i in range(len(self._names))}
        # After this, self._dict maps short names onto empty lists

        print('self._dict', self._dict)

    def _initialise_dict_index(self, names):
        """ Initialises dictionary and index of names. """
        names_short = []
        for name in names:
            num = name.find('.')
            if num < 0:
                names_short.append(name)
            else:
                names_short.append(name[:num])
        # At this point names_short equals a list of parameter names, with
        # vector suffixes stripped. Vector parameters appear multiple times

        names_long = list(names_short)
        # Names long is a copy of names short

        names_short = list(dict.fromkeys(names_short))
        # Names short is now a - randomly ordered - list of unique short names

        index = [names_short.index(name) for name in names_long]
        # Index is now a list of ints, such that each entry corresponds to a
        # parameter (with vector params unrolled). The value of each entry is
        # the index of the corresponding suffix-free parameter name in
        # names_short

        return names_short, index




    def __call__(self, x):
        if self._fit is None:
            raise RuntimeError(
                'No data supplied to create Stan model fit object. '
                'Run `update_data` first.')

        # x is a flat list of parameters, presumably in the order the user
        # expects, i.e. the order in which they appear in the model.
        # For some reason we don't pass this directly to pystan (which also
        # expects this order).

        vals = self._prepare_values(x)
        try:
            print(x)
            print(vals)


            return self._log_prob(vals, adjust_transform=True)
        # if Pints proposes a value outside of Stan's parameter bounds
        except (RuntimeError, ValueError) as e:
            warnings.warn('RuntimeError or ValueError encountered when '
                          'calling `pints.LogPDF`: ' + str(e))
            return -np.inf

    def _prepare_values(self, x):
        """ Flattens lists from PyStan's dictionary. """
        dict = self._dict_update(x)
        # dict is now a dict mapping short param names to values, possibly
        # vector values!

        vals = dict.values()
        # Vals is now an arbitrarily ordered list of values from the dictionary

        print('Prepare order', vals)

        b = []
        for ele in vals:
            if not isinstance(ele, list):
                ele = [ele]
            b.append(ele)
        # b is now a list of vals entries, where the scalars (and n=1 lists)
        # are all stored as lists

        vals = [item for sublist in b for item in sublist]
        # vals is now a flattened version of b

        return vals

    def _dict_update(self, x):
        """ Updates dictionary object with parameter values. """

        # Get short name list
        names = self._names

        k = 0
        for i, name in enumerate(names):

            # Get the size of this parameter (1 for non-vector)
            count = self._counter[i]
            if count == 1:
                # Note that this doesn't distinguish between scalars and length
                # one vectors
                self._dict[name] = x[k]
                k += 1
            else:
                vals = []
                for j in range(count):
                    vals.append(x[k])
                    k += 1
                # This should probably be
                # self._dict[name] = x[k:k + count]
                # k += count

                self._dict[name] = vals
        return self._dict


    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        if self._fit is None:
            raise RuntimeError(
                'No data supplied to create Stan model fit object. '
                'Run `update_data` first.')
        vals = self._prepare_values(x)
        try:
            print(x)
            print(vals)

            val = self._log_prob(vals, adjust_transform=True)
            dp = self._grad_log_prob(vals, adjust_transform=True)
            print(dp.shape, dp.reshape(-1))

            return val, dp.reshape(-1)
        except (RuntimeError, ValueError) as e:
            warnings.warn('RuntimeError or ValueError encountered when '
                          'calling `pints.LogPDF`: ' + str(e))



            #TODO: reshape(-1) doesn't do anything?
            return -np.inf, np.ones(self._n_parameters).reshape(-1)

