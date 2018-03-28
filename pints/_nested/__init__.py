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
import numpy as np


class NestedSampler(object):
    """
    Takes a :class:`LogLikelihood` function and returns a nested sampler.

    Arguments:

    ``log_likelihood``
        A :class:`LogLikelihood` function that evaluates points in the
        parameter space.
    ``log_posterior``
        A :class:`LogPrior` function on the same parameter space.

    """
    def __init__(self, log_likelihood, log_prior):

        # Store log_likelihood and log_prior
        if not isinstance(log_likelihood, pints.LogLikelihood):
            raise ValueError('Given function must extend pints.LogLikelihood')
        self._log_likelihood = log_likelihood

        # Store function
        if not isinstance(log_prior, pints.LogPrior):
            raise ValueError('Given function must extend pints.LogPrior')
        self._log_prior = log_prior

        # Get dimension
        self._dimension = self._log_likelihood.n_parameters()

        # Print info to console
        self._verbose = True

    def run(self):
        """
        Runs the nested sampling routine and returns a tuple of the
        posterior samples and an estimate of the marginal likelihood.
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


def reject_sample_prior(threshold, log_likelihood, log_prior):
    """
    Independently samples params from the prior until
    ``log_likelihood(params) > threshold``.
    """
    proposed = log_prior.sample()[0]
    while log_likelihood(proposed) < threshold:
        proposed = log_prior.sample()[0]
    return np.concatenate((proposed, np.array([log_likelihood(proposed)])))
