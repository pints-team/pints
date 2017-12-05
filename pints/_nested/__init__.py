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

class NestedSampler(object):
    """
    Takes a :class:`LogLikelihood` function and returns a nested sampler.

    Arguments:

    ``function``
        A :class:`LogLikelihood` function that evaluates points in the
        parameter space.

    """
    def __init__(self, log_likelihood, aPrior):

        # Store function
        if not isinstance(log_likelihood, pints.LogLikelihood):
            raise ValueError('Given function must extend pints.LogLikelihood')
        self._log_likelihood = log_likelihood

        # Store function
        if not isinstance(aPrior, pints.Prior):
            raise ValueError('Given function must extend pints.Prior')
        self._prior = aPrior

        # Get dimension
        self._dimension = self._log_likelihood.dimension()

        # Print info to console
        self._verbose = True

    def run(self):
        """
        Runs the nested sampling routine and returns a markov chain representing the
        distribution of the given log-likelihood function.
        """
        raise NotImplementedError

    def set_verbose(self, value):
        """
        Enables or disables verbose mode for this nested sampling routine. In verbose
        mode, lots of output is generated during a run.
        """
        self._verbose = bool(value)

    def verbose(self):
        """
        Returns ``True`` if the nested sampling routine is set to run in verbose mode.
        """
        return self._verbose

