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


class ABCRejection(ABCSampler):
    """
    Rejection ABC.

    Sampling parameters from a user-defined prior distribution, simulating data according to a user-defined model
    comparing against a user-defined threshold, and accepting or rejecting.

    [1] ***Citations Needed***

    """
    def __init__(self, log_prior):

        # Set log_prior
        self._log_prior = log_prior

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Rejection ABC'


    def ask(self, n_samples):
        """ See :meth:`SingleChainMCMC.ask()`. """

        # Sample from prior
        param_vals = self._log_prior.sample(n_samples)

        return param_vals


    def tell(self, xs, fx, threshold):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """
        if len(param_vals) != len(fx):
            print('error: number of parameters must equal number of function outputs')

        accepted_samples = []

        index = np.where(np.array(fx) < threshold)[0]
        accepted_samples.extend(xs[index])

        # for index in range(len(fx)):
           #  if fx[index] < threshold:
           #      accepted_samples.append(param_vals[index])

        return accepted_samples

