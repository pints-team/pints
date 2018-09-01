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


class AdaptiveCovarianceAMMCMC(pints.AdaptiveCovarianceMCMC):
    """
    Adaptive Metropolis MCMC, as described by Algorithm 2 in [1],
    (with gamma = self._adaptations ** -eta which isn't specified
    in the paper)
    
    Initialises mu0 and sigma0 used in proposal N(mu0, lambda * sigma0)
    For iteration t = 0:n_iter:
      - Sample Y_t+1 ~ N(theta_t, lambda * sigma0)
      - Calculate alpha(theta_t, Y_t+1) = min(1, p(Y_t+1|data) / p(theta_t|data))
      - Set theta_t+1 = Y_t+1 with probability alpha(theta_t, Y_t+1); otherwise
      theta_t+1 = theta_t
      - Update mu_t+1 = mu_t + gamma_t+1 * (theta_t+1 - mu_t)
      - Update sigma_t+1 ~ sigma_t + gamma_t+1 * ((theta_t+1 - mu_t)(theta_t+1 - mu_t)' - sigma_t)
    endfor

    [1] A tutorial on adaptive MCMC
    Christophe Andrieu and Johannes Thoms, Statistical Computing,
    2008, 18: 343-373

    *Extends:* :class:`AdaptiveCovarianceMCMC`
    """
    def __init__(self, x0, sigma0=None):
        super(AdaptiveCovarianceRemiMCMC, self).__init__(x0, sigma0)

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """
        super(AdaptiveCovarianceRemiMCMC, self).ask()
        # Propose new point
        if self._proposed is None:

            # Note: Normal distribution is symmetric
            #  N(x|y, sigma) = N(y|x, sigma) so that we can drop the proposal
            #  distribution term from the acceptance criterion
            self._proposed = np.random.multivariate_normal(self._current, np.exp(self._loga) * self._sigma)

            # Set as read-only
            self._proposed.setflags(write=False)

        # Return proposed point
        return self._proposed

    def _initialise(self):
        """
        See :meth: `AdaptiveCovarianceMCMC._initialise()`.
        """
        super(AdaptiveCovarianceRemiMCMC, self)._initialise()
        
        # log adaptation
        self._loga = 0

    def tell(self, fx):
        """ See :meth:`pints.AdaptiveCovarianceMCMC.tell()`. """
        super(AdaptiveCovarianceRemiMCMC, self).tell(fx)
        # Update log acceptance
        if self._adaptive:
            self._loga += self._gamma * (self._accepted - self._target_acceptance)
        # Return new point for chain
        return self._current

