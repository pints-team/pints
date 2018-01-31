#
# Sub-module containing MCMC inference routines
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class MCMC(object):
    """
    Takes a :class:`LogPDF` function and returns a markov chain representative
    of its distribution.

    Arguments:

    ``log_pdf``
        A :class:`LogPDF` function that evaluates points in the parameter
        space.
    ``x0``
        An starting point in the parameter space.
    ``sigma0=None``
        An optional initial covariance matrix, i.e., a guess of the the
        covariance of the ``log_pdf`` around ``x0``.

    """
    def __init__(self, log_pdf, x0, sigma0=None):

        # Store function
        if not isinstance(log_pdf, pints.LogPDF):
            raise ValueError('Given function must extend pints.LogPDF')
        self._log_pdf = log_pdf

        # Get dimension
        self._dimension = self._log_pdf.dimension()

        # Check initial position
        self._x0 = pints.vector(x0)
        if len(self._x0) != self._dimension:
            raise ValueError(
                'Given initial position must have same dimension as function.')

        # Check initial standard deviation
        if sigma0 is None:
            self._sigma0 = np.diag(0.01 * self._x0)
        else:
            self._sigma0 = np.array(sigma0)
            if np.product(self._sigma0.shape) == self._dimension:
                # Convert from 1d array
                self._sigma0 = self._sigma0.reshape((self._dimension,))
                self._sigma0 = np.diag(self._sigma0)
            else:
                # Check if 2d matrix of correct size
                self._sigma0 = self._sigma0.reshape(
                    (self._dimension, self._dimension))
            self._sigma0.setflags(write=False)

        # Print info to console
        self._verbose = True

    def run(self):
        """
        Runs the MCMC routine and returns a markov chain representing the
        distribution of the given log-pdf.
        """
        raise NotImplementedError

    def set_verbose(self, value):
        """
        Enables or disables verbose mode for this MCMC routine. In verbose
        mode, lots of output is generated during a run.
        """
        self._verbose = bool(value)

    def verbose(self):
        """
        Returns ``True`` if the MCMC routine is set to run in verbose mode.
        """
        return self._verbose

