#
# Sub-module containing nested samplers
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints


class NestedSampler(pints.TunableMethod):
    """
    Abstract base class for nested samplers.

    Arguments:

    ``log_likelihood``
        A :class:`LogPDF` function that evaluates points in the parameter
        space.
    ``log_prior``
        A :class:`LogPrior` function on the same parameter space.

    """

    def __init__(self, log_likelihood, log_prior):

        # Store log_likelihood and log_prior
        # if not isinstance(log_likelihood, pints.LogLikelihood):
        if not isinstance(log_likelihood, pints.LogPDF):
            raise ValueError(
                'Given log_likelihood must extend pints.LogLikelihood')
        self._log_likelihood = log_likelihood

        # Store function
        if not isinstance(log_prior, pints.LogPrior):
            raise ValueError('Given log_prior must extend pints.LogPrior')
        self._log_prior = log_prior

        # Get dimension
        self._dimension = self._log_likelihood.n_parameters()
        if self._dimension != self._log_prior.n_parameters():
            raise ValueError(
                'Given log_likelihood and log_prior must have same number of'
                ' parameters.')

        # Logging
        self._log_to_screen = True
        self._log_filename = None
        self._log_csv = False

    def run(self):
        """
        Runs the nested sampling routine and returns a tuple of the
        posterior samples and an estimate of the marginal likelihood.
        """
        raise NotImplementedError

    def set_log_to_file(self, filename=None, csv=False):
        """
        Enables logging to file when a filename is passed in, disables it if
        ``filename`` is ``False`` or ``None``.

        The argument ``csv`` can be set to ``True`` to write the file in comma
        separated value (CSV) format. By default, the file contents will be
        similar to the output on screen.
        """
        if filename:
            self._log_filename = str(filename)
            self._log_csv = True if csv else False
        else:
            self._log_filename = None
            self._log_csv = False

    def set_log_to_screen(self, enabled):
        """
        Enables or disables logging to screen.
        """
        self._log_to_screen = True if enabled else False
