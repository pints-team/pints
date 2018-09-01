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


class AdaptiveCovarianceAMGlobalComponentMCMC(pints.AdaptiveCovarianceMCMC):
    """
    Adaptive Metropolis MCMC, as described by Algorithm 6 in [1],
    (with gamma = self._adaptations ** -eta which isn't specified
    in the paper). The algorithm we use is actually a mixture of Algorithm 6
    and Algorithm 4, as is suggested in the text in [1].
    
    Initialises mu0 and sigma0 used in componentwise proposal N(mu0, lambda * sigma0)

    For iteration t = 0:n_iter:

      if mod(t, self._am_global_rate == 0)
        - Sample Y_t+1 ~ N(theta_t, lambda_t * sigma0)
        - Calculate alpha(theta_t, Y_t+1) = min(1, p(Y_t+1|data) / p(theta_t|data))
        - Update log lambda_t+1^scalar = log lambda_t^scalar +
                                          gamma_t+1 * (alpha(theta_t, Y_t+1) - self._target_acceptance)

      else:
        - Sample Z_t+1 ~ N(0, Lambda^0.5 * sigma0_t * Lambda^0.5)
        - Set Y_t+1 = theta_t + Z_t+1

      endif

      - Set theta_t+1 = Y_t+1 with probability alpha(theta_t, Y_t+1); otherwise
        theta_t+1 = theta_t

        for k in 1:self_.dimensions:
            - Set W_t+1 = zeros(self._dimension)
            - Set W_t+1[k] = Z_t+1[k]
            - log lambda_t+1^k = log lambda_t^k + gamma_t+1 * (alpha(theta_t, theta_t + W_t+1) - self._target_acceptance)
        endfor

      - Update mu_t+1 = mu_t + gamma_t+1 * (theta_t+1 - mu_t)
      - Update sigma_t+1 = sigma_t + gamma_t+1 * ((theta_t+1 - mu_t)(theta_t+1 - mu_t)' - sigma_t)
    endfor

    where e_k is a vector of zeros apart from the kth entry which equals 1;
    lambda_t^k is the kth component of lambda (which is here a vector);
    Lambda_t = diag(lambda_t^1, lambda_t^2, ..., lambda_t^self._dimension);
    lambda_t^scalar is a scalar-valued lambda used in global am

    [1] A tutorial on adaptive MCMC
    Christophe Andrieu and Johannes Thoms, Statistical Computing,
    2008, 18: 343-373

    *Extends:* :class:`AdaptiveCovarianceMCMC`
    """
    def __init__(self, x0, sigma0=None):
        super(AdaptiveCovarianceAMGlobalComponentMCMC, self).__init__(x0, sigma0)

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """
        super(AdaptiveCovarianceAMGlobalComponentMCMC, self).ask()
        # Propose new point
        if self._proposed is None:
            if self._iter_count % self._am_global_rate == 0:
                v_proposed = np.random.multivariate_normal(self._current,
                                                           np.exp(self._log_lambda_scalar) * self._sigma)
            else:
                Lambda = np.diag(np.exp(self._log_lambda_vector))
                Lambda_half = Lambda**0.5
                self._Z = np.random.multivariate_normal(np.zeros(self._dimension),
                                                  np.matmul(np.matmul(Lambda_half, self._sigma),
                                                            Lamda_half))
                v_proposed = self._current + self._Z

            self._proposed = v_proposed

            # Set as read-only
            self._proposed.setflags(write=False)

        # Return proposed point
        return self._proposed

    def _initialise(self):
        """
        See :meth: `AdaptiveCovarianceMCMC._initialise()`.
        """
        super(AdaptiveCovarianceAMGlobalComponentMCMC, self)._initialise()
        self._log_lambda_scalar = 0
        self._log_lambda_vector = np.zeros(self._dimension)
        self._am_global_rate = 10
        self._iter_count = 0
        self._Z = np.zeros(self._dimension)
    
    def set_am_global_rate(self, am_global_rate):
        """
        Sets number of steps between each global am update
        """
        if am_global_rate < 1:
          raise ValueError('Number of steps between global am updates' +
                           ' must exceed 1.')
        if not isinstance(am_global_rate, int):
          raise ValueError('Number of steps between global am updates' +
                           ' must be an integer.')
        self._am_global_rate = am_global_rate

    def tell(self, fx):
        """ See :meth:`pints.AdaptiveCovarianceMCMC.tell()`. """
        super(AdaptiveCovarianceAMGlobalComponentMCMC, self).tell(fx)
        
        if self._iter_count % self._am_global_rate == 0:
            self._log_lambda_scalar += self._gamma * (self._alpha - self._target_acceptance)
        else:
            for k in range(0, self._dimension):
                W = np.zeros(self._dimension)
                W[k] = Z[k]
                ## This line is problematic for ask/tell since
                ## alpha = min(1, p(self._X + W|data) / p(self._X|data))
                self._log_lambda_vector[k] += self._gamma * (alpha(self._X, self._X + W) - self._target_acceptance)
        self._iter_count += 1

        # Return new point for chain
        return self._current
