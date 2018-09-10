#
# Nested rejection sampler implementation.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
from scipy.misc import logsumexp


class NestedRejectionSampler(pints.NestedSampler):
    """
    Creates a nested sampler that estimates the marginal likelihood
    and generates samples from the posterior.

    This is the simplest form of nested sampler and involves using
    rejection sampling from the prior as described in the algorithm on page 839
    in [1] to estimate the marginal likelihood and generate weights,
    preliminary samples (with their respective likelihoods), required to
    generate posterior samples.

    The posterior samples are generated as described in [1] on page 849 by
    randomly sampling the preliminary point, accounting for their weights and
    likelihoods.

    *Extends:* :class:`NestedSampler`

    [1] "Nested Sampling for General Bayesian Computation", John Skilling,
    Bayesian Analysis 1:4 (2006).
    """

    def __init__(self, log_likelihood, log_prior):
        super(NestedRejectionSampler, self).__init__(log_likelihood, log_prior)

    def run(self):
        """ See :meth:`pints.NestedSampler.run()`. """
        super(NestedRejectionSampler, self).run()

    def ask(self):
        """
        Proposes a new point by sampling from the prior
        """
        return self._log_prior.sample()[0]

