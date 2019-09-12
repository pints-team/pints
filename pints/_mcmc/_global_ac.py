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


class GlobalACMCMC(pints.GlobalAdaptiveCovarianceMCMC):
    """
    Adaptive Metropolis MCMC, as described by Algorithm 4 in [1],
    (with gamma = adaptation_count^-eta which isn't specified
    in the paper).

    Initialises::

        mu
        Sigma
        adaptation_count = 0

    In each adaptive iteration (t)::

        Sample theta* ~ N(theta_t, lambda * Sigma)
        alpha = min(1, p(theta*|data) / p(theta_t|data))
        u ~ uniform(0, 1)
        if alpha > u:
            theta_(t+1) = theta*
        else:
            theta_(t+1) = theta_t
        mu += gamma * (theta_(t+1) - mu)
        Sigma += gamma * ((theta_(t+1) - mu)(theta_(t+1) - mu) - Sigma)
        log lambda += gamma * (alpha - self._target_acceptance)
        gamma = adaptation_count^-eta

    [1] A tutorial on adaptive MCMC
    Christophe Andrieu and Johannes Thoms, Statistical Computing,
    2008, 18: 343-373

    *Extends:* :class:`AdaptiveCovarianceMCMC`
    """
    def __init__(self, x0, sigma0=None):
        super(GlobalACMCMC, self).__init__(x0, sigma0)

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """
        super(GlobalACMCMC, self).ask()

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

    def _initialise(self):
        """
        See :meth: `AdaptiveCovarianceMCMC._initialise()`.
        """
        super(GlobalACMCMC, self)._initialise()
        self._log_lambda = 0

    def tell(self, fx):
        """ See :meth:`pints.AdaptiveCovarianceMCMC.tell()`. """
        super(GlobalACMCMC, self).tell(fx)

        self._alpha = np.minimum(1, np.exp(self._r))

        self._log_lambda += (self._gamma *
                             (self._alpha - self._target_acceptance))

        # Return new point for chain
        return self._current
