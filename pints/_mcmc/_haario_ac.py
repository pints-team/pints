#
# Adaptive covariance MCMC method
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


class HaarioACMC(pints.GlobalAdaptiveCovarianceMCMC):
    """
    Adaptive Metropolis MCMC, which is algorithm 4 in [1] and is described in
    the text in [2]. Differs from ``HaarioBardenetACMC`` only through its use
    of ``alpha`` in the updating of ``log_lambda`` (rather than a binary
    accept/reject).

    Initialise::

        mu
        Sigma
        adaptation_count = 0
        log lambda = 0

    In each adaptive iteration (t)::

        adaptation_count = adaptation_count + 1
        gamma = (adaptation_count)^-eta
        theta* ~ N(theta_t, lambda * Sigma)
        alpha = min(1, p(theta*|data) / p(theta_t|data))
        u ~ uniform(0, 1)
        if alpha > u:
            theta_(t+1) = theta*
            accepted = 1
        else:
            theta_(t+1) = theta_t
            accepted = 0

        mu = (1 - gamma) mu + gamma theta_(t+1)
        Sigma = (1 - gamma) Sigma + gamma (theta_(t+1) - mu)(theta_(t+1) - mu)
        log lambda = log lambda + gamma (alpha - self._target_acceptance)
        gamma = adaptation_count^-eta

    [1] A tutorial on adaptive MCMC
    Christophe Andrieu and Johannes Thoms, Statistical Computing, 2008,
    18: 343-373.

    [2] An adaptive Metropolis algorithm
    Heikki Haario, Eero Saksman, and Johanna Tamminen (2001) Bernoulli.

    *Extends:* :class:`GlobalAdaptiveCovarianceMCMC`
    """
    def __init__(self, x0, sigma0=None):
        super(HaarioACMC, self).__init__(x0, sigma0)
        self._log_lambda = 0
        self._binary_accept = True
        self._accepted = True

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """
        super(HaarioACMC, self).ask()

        # Propose new point
        if self._proposed is None:
            self._proposed = (
                np.random.multivariate_normal(self._current,
                                              ((np.exp(self._log_lambda) *
                                               self._sigma)))
            )

            # Set as read-only
            self._proposed.setflags(write=False)

        # Return proposed point
        return self._proposed

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 1

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Haario adaptive covariance MCMC'

    def tell(self, fx):
        """ See :meth:`pints.AdaptiveCovarianceMCMC.tell()`. """
        super(HaarioACMC, self).tell(fx)

        _acceptance_prob = (
            np.minimum(1, np.exp(self._log_acceptance_ratio)))
        if self._adaptive:
            self._log_lambda += (self._gamma *
                                 (_acceptance_prob -
                                  self._target_acceptance))

        # Return new point for chain
        return self._current
