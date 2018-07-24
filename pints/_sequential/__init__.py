#
# Sub-module containing sequential MC inference routines
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


class SMCSampler(object):
    """
    Takes a :class:`LogPosterior` function and returns a SMC sampler.

    Arguments:

    ``log_posterior``
        A :class:`LogPosterior` function that evaluates points in the
        parameter space.

    """
    def __init__(self, log_posterior, x0, sigma0=None):

        # Store log_likelihood and log_prior
        if not isinstance(log_posterior, pints.LogPDF):
            raise ValueError('Given function must extend pints.LogPDF')
        self._log_posterior = log_posterior
        
        # Check initial position
        self._x0 = pints.vector(x0)

        # Get dimension
        self._dimension = len(self._x0)
        
        # Check initial standard deviation
        if sigma0 is None:
            # Get representative parameter value for each parameter
            self._sigma0 = np.abs(self._x0)
            self._sigma0[self._sigma0 == 0] = 1
            # Use to create diagonal matrix
            self._sigma0 = np.diag(0.01 * self._sigma0)
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

        # Get dimension
        self._dimension = self._log_posterior.n_parameters()

        # Print info to console
        self._verbose = True

    def run(self):
        """
        Runs the SMC sampling routine and returns a tuple of the
        posterior samples.
        """
        raise NotImplementedError

    def set_verbose(self, value):
        """
        Enables or disables verbose mode for this nested sampling routine. In
        verbose mode, lots of output is generated during a run.
        """
        self._verbose = bool(value)

    def verbose(self):
        """
        Returns ``True`` if the nested sampling routine is set to run in
        verbose mode.
        """
        return self._verbose
