#
# Base class for Adaptive covariance MCMC methods
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
from scipy.misc import logsumexp
import scipy


class LocalAdaptiveCovarianceMCMC(pints.AdaptiveCovarianceMCMC):
    """
    Base class for single chain MCMC methods that adapt a covariance matrix
    when running, in order to control the acceptance rate. This class is for
    those ACMCMC algorithms that update a local proposal kernel.

    In all cases ``self._adaptations ^ -eta`` is used to control decay of
    adaptation

    *Extends:* :class:`SingleChainMCMC`
    """
    def __init__(self, x0, sigma0=None):
        super(LocalAdaptiveCovarianceMCMC, self).__init__(x0, sigma0)
        # Localised AM
        self._initial_fit = True
        self._Z = 0

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """
        super(LocalAdaptiveCovarianceMCMC, self).ask()

    def tell(self, fx):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """
        super(LocalAdaptiveCovarianceMCMC, self).tell(fx)

        self._r += self._ratio_q()

        self._alpha = np.minimum(1, np.exp(self._r))
        self._accepted = 0
        if np.isfinite(fx):
            u = np.log(np.random.uniform(0, 1))
            if u < self._r:
                self._accepted = 1
                self._current = self._proposed
                self._current_log_pdf = fx

        # Clear proposal
        self._proposed = None

        # Update acceptance rate (only used for output!)
        self._acceptance = (
            (self._iterations * self._acceptance + self._accepted)
            / (self._iterations + 1))

        # Increase iteration count
        self._iterations += 1

        # Adapt covariance matrix
        if self._adaptive:
            # Set gamma based on number of adaptive iterations
            self._gamma = self._adaptations ** -self._eta
            self._adaptations += 1

            self._update_gaussian_mixture()
            self._initial_fit = False
        return self._current

    def _update_gaussian_mixture(self):
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
                                scipy.stats.multivariate_normal.logpdf(
                                self._current,
                                self._mu[i], self._sigma[i],
                                allow_singular=True))
        self._log_q_l += -logsumexp(self._log_q_l)

    def _update_mu(self):
        """
        Updates mu components according to,
        mu_t+1^k = mu_t^k + gamma_t+1 * q_t^k * (theta_t+1 - mu_t^k)
        """
        for i in range(self._mixture_components):
            self._mu[i] = ((1 - self._gamma * np.exp(self._log_q_l[i]))
                           * self._mu[i] +
                           self._gamma * np.exp(self._log_q_l[i])
                           * self._current)

    def _update_sigma(self):
        """
        Updates sigma components according to,
        sigma_t+1^k = sigma_t^k + gamma_t+1 * q_t^k *
                      ((theta_t+1 - mu_t^k)(theta_t+1 - mu_t^k)' - sigma_t^k)
        """
        for i in range(self._mixture_components):
            dsigm = np.reshape(self._current -
                               self._mu[i], (self._dimension, 1))
            self._sigma[i] = self._sigma[i] + self._gamma * (
                np.exp(self._log_q_l[i]) * (
                    np.dot(dsigm, dsigm.T) - self._sigma[i]))

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
            q_numerator.append(
                np.log(self._w[i]) +
                scipy.stats.multivariate_normal.logpdf(self._Y,
                                                       self._mu[i],
                                                       self._sigma[i],
                                                       allow_singular=True))
            q_denominator.append(
                np.log(self._w[i]) +
                scipy.stats.multivariate_normal.logpdf(self._X,
                                                       self._mu[i],
                                                       self._sigma[i],
                                                       allow_singular=True))
        q_numerator = q_numerator - logsumexp(q_numerator)
        q_denominator = q_denominator - logsumexp(q_denominator)

        # mistake in the paper!
        return q_numerator[self._Z] - q_denominator[self._Z]
