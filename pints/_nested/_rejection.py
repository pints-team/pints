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
        super(NestedRejectionSampler)

    def set_active_points_rate(self, active_points):
        """
        Sets the number of active points for the next run.
        """
        active_points = int(active_points)
        if active_points <= 5:
            raise ValueError('Number of active points must be greater than 5.')
        self._active_points = active_points

    def n_hyper_parameters(self):
        """
        Returns the number of hyper-parameters for this method (see
        :class:`TunableMethod`).
        """
        return 1

    def set_hyper_parameters(self, x):
        """
        Sets the hyper-parameters for the method with the given vector of
        values (see :class:`TunableMethod`).

        Hyper-parameter vector is: ``[active_points_rate]``

        Arguments:

        ``x`` an array of length ``n_hyper_parameters`` used to set the
              hyper-parameters
        """

        self.set_active_points_rate(x[0])

    def set_iterations(self, iterations):
        """
        Sets the total number of iterations to be performed in the next run.
        """
        iterations = int(iterations)
        if iterations < 0:
            raise ValueError('Number of iterations cannot be negative.')
        self._iterations = iterations

    def set_posterior_samples(self, posterior_samples):
        """
        Sets the number of posterior samples to generate from points proposed
        by the nested sampling algorithm.
        """
        posterior_samples = int(posterior_samples)
        if posterior_samples < 1:
            raise ValueError(
                'Number of posterior samples must be greater than zero.')
        self._posterior_samples = posterior_samples
