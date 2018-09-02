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


class AdaptiveCovarianceLocalisedMCMC(pints.AdaptiveCovarianceMCMC):
    """
    Adaptive Metropolis MCMC, as described by Algorithm 7 in [1],
    (with gamma = self._adaptations ** -eta which isn't specified
    in the paper)
    
    This algorthm has n possible proposal distributions, where the
    different proposals are chosen dependent on location in parameter
    space.
    
    Algorithm:
    
    Based on initial unadaptive samples, fit Gaussian mixture model
    to samples and obtain w^n, mu^n and sigma^n
    
    Initialise lambda^1:n
    
    For iteration t = 0:n_iter:
      - Sample Z_t+1 ~ categorical(q_weight(1, theta_t), q_weight(2, theta_t),..., q_weight(n, theta_t))
      - Sample Y_t+1 ~ N(theta_t, lambda_t^Z_t+1 sigma_t^Z_t+1)
      - Set theta_t+1 = Y_t+1 with probability,
        min(1, [p(Y_t+1|data) * q_weight(Y_t+1|Z_t+1) /
               [p(Y_t+1|data) *q_weight(theta_t|Z_t+1)]);
        otherwise theta_t+1 = theta_t
      - Update mu^1:n, sigma^1:n, w^1:n and lambda^1:n as shown below
    endfor
    
    w^1:n are the weights of the different normals in fitting
    q(theta) = sum_i=1^n w^k N(theta|mu^k, sigma^k) to samples;
    q_weight(theta|kk) = w^k N(theta|mu^k, sigma^k) / q(theta).
    
    The update steps are as follows,
    
    for k in 1:n
        - Calculate Q = q(theta_t+1|k)
        - mu_t+1^k = mu_t^k + gamma_t+1 * Q * (X_t+1 - mu_t^k)
        - sigma_t+1^k = sigma_t^k + gamma_t+1 * Q * ((theta_t+1 - mu_t)(theta_t+1 - mu_t)' - sigma_t^k)
        - w_t+1^k = w_t^k + gamma_t+1 * (Q - w_t^k)
    endfor

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
        self._localised = True
        self._mixture_components = 3
        
        # Initialise weights
        self._w = np.repeat(1.0 / self._mixture_components,
                            self._mixture_components)
        # Create lists of means and covariance matrices
        self._mu = [self._mu for i in range(self._mixture_components)]
        self._sigma = [self._sigma for i in range(self._mixture_components)]
        
        # Initialise lambda vector
        self._log_lambda = np.zeros(self._mixture_components)
        
        # Initialise running expected acceptance probabilities
        self._alpha = np.zeros(self._mixture_components)

    def set_mixture_components(self, mixture_components):
        """
        Sets the number of Gaussian mixture components
        to use for proposals
        """
        if mixture_components < 2:
            raise ValueError('Number of mixture components ' +
                             'should exceed 1.')
        if not isinstance(mixture_components, int):
            raise ValueError('Number of mixture components ' +
                             'should be an integer.')
        self._mixture_components = mixture_components

    def tell(self, fx):
        """ See :meth:`pints.AdaptiveCovarianceMCMC.tell()`. """
        super(AdaptiveCovarianceLocalisedMCMC, self).tell(fx)
        
        self._log_lambda += self._gamma * (self._alpha - self._target_acceptance)
        
        # Return new point for chain
        return self._current
        
    def _fit_gaussian_mixture(self):
        """
        See :meth: `AdaptiveCovarianceMCMC.__fit_gaussian_mixture()`
        """
        self.update_q()
        self.update_mu()
        self.update_sigma()
        self.update_w()
        self.update_lambda()
        self.update_alpha()
        
    def update_q(self):
        """
        Updates q functions representing weights of Gaussian mixture
        components. If first time this is called, then
        this function creates q functions
        """
        
            
