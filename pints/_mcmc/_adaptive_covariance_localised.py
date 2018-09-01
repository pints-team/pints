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
from sklearn import mixture


class AdaptiveCovarianceLocalisedMCMC(pints.AdaptiveCovarianceMCMC):
    """
    Adaptive Metropolis MCMC, as described by Algorithm 7 in [1],
    (with gamma = self._adaptations ** -eta which isn't specified
    in the paper)
    
    This algorthm has n possible proposal distributions, where the
    different proposals are chosen dependent on location in parameter
    space.
    
    Initialise mu0^1:n, sigma0^1:n, w^1:n and lambda^1:n
    
    For iteration t = 0:n_iter:
      - Fit Gaussian mixture model to (theta_0,theta_1,theta_t) and
        obtain weights w of each of n modes. Also obtain mu^n and
        sigma^n for each proposal
      - Sample Z_t+1 ~ categorical(w)
      - Sample Y_t+1 ~ N(theta_t, lambda_t^Z_t+1 sigma_t^Z_t+1)
      - Set theta_t+1 = Y_t+1 with probability,
        min(1, [p(Y_t+1|data) * / )
    endfor
    
    w^1:n are the weights of the different normals in fitting
    q(theta) = sum_i=1^n w^k N(theta|mu^k, sigma^k) to previous
    theta samples

    [1] A tutorial on adaptive MCMC
    Christophe Andrieu and Johannes Thoms, Statistical Computing,
    2008, 18: 343-373

    *Extends:* :class:`AdaptiveCovarianceMCMC`
    """
    def __init__(self, x0, sigma0=None):
        super(AdaptiveCovarianceLocalisedMCMC, self).__init__(x0, sigma0)

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """
        super(AdaptiveCovarianceLocalisedMCMC, self).ask()
        # Propose new point
        if self._proposed is None:
            self._proposed = np.random.multivariate_normal(self._current,
                                                           np.exp(self._log_lambda) * self._sigma)

            # Set as read-only
            self._proposed.setflags(write=False)

        # Return proposed point
        return self._proposed

    def _initialise(self):
        """
        See :meth: `AdaptiveCovarianceMCMC._initialise()`.
        """
        super(AdaptiveCovarianceLocalisedMCMC, self)._initialise()
        self._log_lambda = 0

    def tell(self, fx):
        """ See :meth:`pints.AdaptiveCovarianceMCMC.tell()`. """
        super(AdaptiveCovarianceLocalisedMCMC, self).tell(fx)
        
        self._log_lambda += self._gamma * (self._alpha - self._target_acceptance)
        
        # Return new point for chain
        return self._current
