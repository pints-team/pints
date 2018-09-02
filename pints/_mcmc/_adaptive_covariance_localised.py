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
import scipy
from scipy.misc import logsumexp


class AdaptiveCovarianceLocalisedMCMC(pints.AdaptiveCovarianceMCMC):
    """
    Adaptive Metropolis MCMC, as described by Algorithm 7 in [1],
    (with gamma = self._adaptations ** -eta which isn't specified
    in the paper). Note think there is a mistake in this algorithm
    in calculating the ratio. It shoud be q(k|X_t) / p(k|Y_t+1)
    rather than the reverse; otherwise the algorithm doesn't work.
    
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
      - Set theta_t+1 = Y_t+1 with probability alpha_t^Z_t+1(theta_t, Y_t+1)
        otherwise theta_t+1 = theta_t
      - Update mu^1:n, sigma^1:n, w^1:n and lambda^1:n as shown below
    endfor
    
    w^1:n are the weights of the different normals in fitting
    q(theta) = sum_i=1^n w^k N(theta|mu^k, sigma^k) to samples;
    q_weight(theta|kk) = w^k N(theta|mu^k, sigma^k) / q(theta).
    alpha_t^k = min(1, [p(Y_t+1|data) * q_weight(Y_t+1|k) /
               [p(Y_t+1|data) *q_weight(theta_t|k)]);
    
    The update steps are as follows,
    
    for k in 1:n
        - Calculate Q = q(theta_t+1|k)
        - mu_t+1^k = mu_t^k + gamma_t+1 * Q * (X_t+1 - mu_t^k)
        - sigma_t+1^k = sigma_t^k + gamma_t+1 * Q * ((theta_t+1 - mu_t^k)(theta_t+1 - mu_t^k)' - sigma_t^k)
        - w_t+1^k = w_t^k + gamma_t+1 * (Q - w_t^k)
        - log lambda_t+1^k = log lambda_t^k + gamma_t+1 * 1(Z_t+1==k?) *
                             (alpha_k(theta_t, Y_t+1) - self._target_acceptance)
        - alpha_t+1^k = alpha_t^k + gamma_t+1 * 1(Z_t+1==k?) *
                             (alpha_k(theta_t, Y_t+1) - self._target_acceptance)
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
            # Sample Z from categorical distribution over q
            self._Z = np.random.choice(self._mixture_components,
                                       size=1, p=np.exp(self._log_q_l).tolist())[0]
            # choose proposed value from Zth proposal distribution
            self._proposed = np.random.multivariate_normal(self._current,
                                                           np.exp(self._log_lambda[self._Z])
                                                           * self._sigma[self._Z])
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

        # Create lists of randomised means and covariance matrices
        epsilon_mu = np.random.normal(0.01 * self._mu, 1, size=(self._mixture_components,
                                      self._dimension))
        epsilon_sigma_v = np.random.normal(np.zeros(self._dimension),
                                           0.01,
                                           size=(self._mixture_components,
                                                 self._dimension))
        self._epsilon_sigma = []
        for i in range(self._mixture_components):
            dsigm = np.reshape(epsilon_sigma_v[i], (self._dimension, 1))
            self._epsilon_sigma.append(np.dot(dsigm, dsigm.T))
        a_temp = np.copy(self._mu)
        a_temp_sigma = np.copy(self._sigma)
        self._mu = []
        self._sigma = []
        for i in range(self._mixture_components):
            self._mu.append(a_temp + epsilon_mu[i])
            self._sigma.append(a_temp_sigma + self._epsilon_sigma[i])

        # Initialise lambda vector
        self._log_lambda = np.zeros(self._mixture_components)
        
        # Initialise running expected acceptance probabilities
        self._alpha_l = np.zeros(self._mixture_components)
        
        # Initialise log_q_l
        self._log_q_l = np.log(np.repeat(1.0 / self._mixture_components,
                                         self._mixture_components))

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
        
        # Return new point for chain
        return self._current
        
    def _fit_gaussian_mixture(self):
        """
        Fits a Gaussian mixture distribution by updating
        componentwise the means, covariance matrices,
        weights and lamdas (eq. 36 and 37 in Andrieu &
        Thoms 2008)
        """
        self._update_log_q()
        self._update_mu()
        self._update_sigma()
        self._update_w()
        self._update_lambda()
        self._update_alpha()
        
    def _update_log_q(self):
        """
        Updates log q values representing weights of Gaussian mixture
        components. If first time this is called, then
        this function creates q functions
        """
        self._log_q_l = np.zeros(self._mixture_components)
        for i in range(self._mixture_components):
            self._log_q_l[i] = (np.log(self._w[i]) +
                                   scipy.stats.multivariate_normal.logpdf(self._current,
                                                         self._mu[i], self._sigma[i],
                                                         allow_singular=True))
        self._log_q_l += -logsumexp(self._log_q_l)

    def _update_mu(self):
        """
        Updates mu components according to,
        mu_t+1^k = mu_t^k + gamma_t+1 * q_t^k * (theta_t+1 - mu_t^k)
        """
        for i in range(self._mixture_components):
            self._mu[i]  = ((1 - self._gamma * np.exp(self._log_q_l[i])) * self._mu[i] +
                            self._gamma * np.exp(self._log_q_l[i]) * self._current)

    def _update_sigma(self):
        """
        Updates sigma components according to,
        sigma_t+1^k = sigma_t^k + gamma_t+1 * q_t^k *
                      ((theta_t+1 - mu_t^k)(theta_t+1 - mu_t^k)' - sigma_t^k)
        """
        for i in range(self._mixture_components):
            dsigm = np.reshape(self._current - self._mu[i], (self._dimension, 1))
            self._sigma[i] = self._sigma[i] + self._gamma * np.exp(self._log_q_l[i]) * (
                             np.dot(dsigm, dsigm.T) - self._sigma[i])

    def _update_w(self):
        """
        Updates w components according to,
        w_t+1^k = w_t^k + gamma_t+1 * (q_t^k - w_t^k)
        """
        for i in range(self._mixture_components):
            self._w[i] += self._gamma * (np.exp(self._log_q_l[i]) - self._w[i])

    def _update_lambda(self):
        """
        Updates lambda components according to,
        log lambda_t+1^k = log lambda_t^k + gamma_t+1 * 1(Z_t+1==k?) *
                             (alpha_k(theta_t, Y_t+1) - self._target_acceptance)
        """
        # Only update Zth component
        self._log_lambda[self._Z] += (self._gamma *
                                     (self._alpha - self._target_acceptance))
      
    def _update_alpha(self):
        """
        Updates running acceptance probabilities according to,
        alpha_t+1^k = alpha_t^k + gamma_t+1 * 1(Z_t+1==k?) * 
                             (alpha_k(theta_t, Y_t+1) - alpha_t^k)
        """
        self._alpha_l[self._Z] += (self._gamma *
                                     (self._alpha - self._alpha_l[self._Z]))

    def _ratio_q(self):
        """
        Yields log q(Y_t+1|Z_t+1) - log q(theta_t|Z_t+1)
        """
        q_numerator = []
        q_denominator = []
        for i in range(self._mixture_components):
            q_numerator.append(np.log(self._w[i]) + 
              scipy.stats.multivariate_normal.logpdf(self._Y,
                                                self._mu[i],
                                                self._sigma[i],allow_singular=True))
            q_denominator.append(np.log(self._w[i]) + 
              scipy.stats.multivariate_normal.logpdf(self._X,
                                                self._mu[i],
                                                self._sigma[i],allow_singular=True))
        q_numerator = q_numerator - logsumexp(q_numerator)
        q_denominator = q_denominator - logsumexp(q_denominator)

        # mistake in the paper!
        return q_numerator[self._Z] - q_denominator[self._Z]
