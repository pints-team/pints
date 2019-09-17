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


class NestedRejectionSampler(pints.NestedSampler):
    """
    Creates a nested sampler that estimates the marginal likelihood
    and generates samples from the posterior.

    This is the simplest form of nested sampler and involves using
    rejection sampling from the prior as described in the algorithm on page 839
    in [1] to estimate the marginal likelihood. In doing so,
    this algorithm also generates posterior samples as a by-product.
    The algorithm is given by the following steps:

    Initialise::

        Z = 0
        X_0 = 1

    Draw samples from prior::

        for i in 1:n_active_points:
            theta_i ~ p(theta), i.e. sample from the prior
            L_i = p(theta_i|X)
        endfor

    In each iteration of the algorithm (t)::

        L_min = min(L)
        indexmin = min_index(L)
        X_t = exp(-t / n_active_points)
        w_t = X_t - X_t-1
        Z = Z + L_min * w_t
        theta* ~ p(theta)
        while p(theta*|X) < L_min:
            theta* ~ p(theta)
        endwhile
        theta_indexmin = theta*
        L_indexmin = p(theta*|X)

    At the end of iterations, there is a final ``Z`` increment::

        Z = Z + (1 / n_active_points) * (L_1 + L_2 + ..., + L_n_active_points)

    The posterior samples are generated as described in [1] on page 849 by
    weighting each dropped sample in proportion to the volume of the
    posterior region it was sampled from. That is, the probability
    for drawing a given sample j is given by::

        p_j = L_j * w_j / Z

    where j = 1, ..., n_iterations.

    *Extends:* :class:`NestedSampler`

    [1] "Nested Sampling for General Bayesian Computation", John Skilling,
    Bayesian Analysis 1:4 (2006).
    """
    def __init__(self, log_prior):
        super(NestedRejectionSampler, self).__init__(log_prior)

        self._needs_sensitivities = False

    def ask(self):
        """
        Proposes new point(s) by sampling from the prior.
        """
        self._proposed = self._log_prior.sample()[0]
        return self._proposed

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 1

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[n_active_points]``

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_n_active_points(x[0])

    def name(self):
        """ See :meth:`pints.NestedSampler.name()`. """
        return 'Nested rejection sampler'
