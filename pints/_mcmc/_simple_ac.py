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


class SimpleACMCMC(pints.GlobalAdaptiveCovarianceMCMC):
    """
    Adaptive Metropolis MCMC, which is either algorithm 4 in [1]
    (if ``binary_accept`` is false) or the algorithm in SOM of [2]
    (if ``binary_accept`` is true). By default, ``binary_accept`` is true.

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

        if binary_accept = 1:
            alpha = accepted

        mu = (1 - gamma) mu + gamma theta_(t+1)
        Sigma = (1 - gamma) Sigma + gamma (theta_(t+1) - mu)(theta_(t+1) - mu)
        log lambda = log lambda + gamma (alpha - self._target_acceptance)
        gamma = adaptation_count^-eta

    [1] A tutorial on adaptive MCMC
    Christophe Andrieu and Johannes Thoms, Statistical Computing,
    2008, 18: 343-373

    [2] Uncertainty and variability in models of the cardiac action potential:
    Can we build trustworthy models?
    Johnstone, Chang, Bardenet, de Boer, Gavaghan, Pathmanathan, Clayton,
    Mirams (2015) Journal of Molecular and Cellular Cardiology

    *Extends:* :class:`AdaptiveCovarianceMCMC`
    """
    def __init__(self, x0, sigma0=None):
        super(SimpleACMCMC, self).__init__(x0, sigma0)
        self._log_lambda = 0
        self._binary_accept = True
        self._accepted = True

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """
        super(SimpleACMCMC, self).ask()

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

    def binary_accept(self):
        """
        Returns binary_accept which determines whether algorithm [2]
        (if binary_accept is true) or algorithm [1] (if binary_accept is false)
        is used.
        """
        return self._binary_accept

    def set_binary_accept(self, binary_accept):
        """
        Determines whether algorithm [2] (if binary_accept is true) or
        algorithm [1] (if binary_accept is false) is used.
        """
        self._binary_accept = bool(binary_accept)

    def tell(self, fx):
        """ See :meth:`pints.AdaptiveCovarianceMCMC.tell()`. """
        super(SimpleACMCMC, self).tell(fx)

        if self._binary_accept:
            self._acceptance_prob = self._accepted
        else:
            self._acceptance_prob = (
                np.minimum(1, np.exp(self._log_acceptance_ratio)))
        self._log_lambda += (self._gamma *
                             (self._acceptance_prob - self._target_acceptance))

        # Return new point for chain
        return self._current
