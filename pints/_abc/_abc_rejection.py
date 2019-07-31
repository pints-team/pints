#
# Adaptive covariance MCMC method
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class ABCRejection(pints.ABCSampler):
    """
    ABC Rejection.

    Sampling parameters from a user-defined prior distribution, simulating data
    according to a user-defined model comparing against a user-defined
    threshold, and accepting or rejecting.

    [1] ***Citations Needed***
    """
    def __init__(self, log_prior, threshold):

        # Set log_prior
        self._log_prior = log_prior

        # Set threshold
        self._threshold = threshold

        # Initialize param_vals
        self._xs = None

    def name(self):
        """ See :meth:`pints.ABCSampler.name()`. """
        return 'Rejection ABC'

    def ask(self, n_samples):
        """ See :meth:`ABCSampler.ask()`. """

        # Sample from prior
        param_vals = self._log_prior.sample(n_samples)
        self._xs = np.copy(param_vals)

        return param_vals

    def tell(self, fx):
        """ See :meth:`ABCSampler.tell()`. """
        if len(self._xs) != len(fx):
            raise ValueError('number of parameters must equal number of '
                             'function outputs')

        accepted_samples = []

        index = np.where(np.array(fx) < self._threshold)[0]
        accepted_samples.extend(self._xs[index])

        return accepted_samples

