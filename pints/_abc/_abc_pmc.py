#
# Population Monte Carlo Approximate Bayesian Computation
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import numpy as np

# TODO: make this work for high dimensionality
class ABCPMC(pints.ABCSampler):
    r"""
    Implements the population monte carlo ABC as described in [1].

    Here is a high-level description of the algorithm:

        for i = 1 to N do:
            repeat
                Sample :math:`\theta_i^{(1)} \sim \pi(\theta)`
                Simulate :math:`x \sim f(x | \theta_i^{(1)})`
            until :math:`\ro(S(x), S(y))`
    
    TODO: finish high-level description + explanation

    References
    ----------
    .. [1] "Adaptive approximate Bayesian computation". Beaumont, M. A.,
           Cornuet, J. M., Marin, J. M., & Robert, C. P. (2009).
           Biometrika, 96(4), 983-990.
           https://doi.org/10.1093/biomet/asp052
    """
    def __init__(self, log_prior):

        self._log_prior = log_prior
        self._threshold = 1
        self._xs = None
        self._ready_for_tell = False
        self._weights = np.array([])
        self._eps = 1
        # TODO: make this a hyperparameter
        self._T = 500
        self._t = 1
        self._i = 0

    def name(self):
        """ See :meth:`pints.ABCSampler.name()`. """
        return 'Rejection ABC'
    
    def psi(self, x):
        """ Evaluates the psi function. """
        return np.exp(-(x ** 2) / 2) / np.sqrt(2 * np.pi)
    
    def emp_var(self):
        """ Computes the weighted empirical variance of self._theta. """
        # Compute weighted mean
        w_mean = 0.0
        for i in range(self._N):
            w_mean = w_mean + self._weights[i] * self._theta[i]
        
        # Compute the sum of the weights
        w_sum = 0.0
        for i in range(self._N):
            w_sum = w_sum + self._weights[i]
        
        # Compute sum of the squared weights
        w_sq_sum = 0.0
        for i in range(self._N):
            w_sq_sum = w_sq_sum + (self._weights[i] ** 2)

        # Compute the non-corrected variance estimation
        n_V = 0.0
        for i in range(self._N):
            n_V = n_V + self.weights[i] * ( (self._theta[i] - w_mean) ** 2 )

        # Add correction term
        e_var = (w_sum ** 2) / ((w_sum ** 2) - w_sq_sum) * n_V

        return e_var

    def ask(self, n_samples):
        """ See :meth:`ABCSampler.ask()`. """
        if self._ready_for_tell:
            raise RuntimeError('Ask called before tell.')
        
        self._ready_for_tell = True

        if self._t == 1:
            if self._i == 0:
                # Initialize variables dependent on N
                self._N = n_samples
                self._i = 1
                self._theta = np.zeros(shape=(1, self._N))
                self._n_theta = np.zeros(shape=(1, self._N))
                self._xs = self._log_prior.sample(n_samples)
                for i in range(n_samples):
                    self._weights = np.append(self._weights, 1 / n_samples)

            # Sample theta_i 
            self._xs = self._log_prior.sample(1)
        else:
            # Sample theta_star
            pt = np.random.uniform()
            theta_star = 0
            partial_sum = 0
            for i in range(self._N):
                if theta_star == 0 and pt <= partial_sum + self._weights[i]:
                    theta_star = self._theta[i]
            
            # Generate sample
            self._n_theta[i] = np.random.normal(theta_star, self._cov)
            self._xs = self._n_theta[i]

        return self._xs

    def tell(self, fx):
        """ See :meth:`ABCSampler.tell()`. """
        if not self._ready_for_tell:
            raise RuntimeError('Tell called before ask.')
        self._ready_for_tell = False

        if self._t == 1:
            # Received 
            if fx < self._eps:
                # Write the definite value of theta_i^t
                self._theta[self._i] = self._xs
                # Increase i or t
                if self._i == self._N:
                    # Also update the covariance
                    self._cov = 2 * self.emp_var()
                    self._i = 1
                    self._t = self._t + 1
                else:
                    self._i = self._i + 1
        else:
            if fx < self._eps:
                self._n_theta[self._i] = self._xs
                if self._i == self._N and self._t == self._T:
                    # Finished
                    return self._n_theta
                else:
                    # Update weight i
                    norm_term = 0.0
                    for i in range(self._N):
                        norm_term = norm_term + self._weights[i] * self.psi((self._n_theta[i] - self._theta[i]) / self._cov) / self._cov

                    self._n_weights[i] = (self._log_prior(self._n_theta[i]) / norm_term)
                    if self._i == self._N:
                        self._i = 1
                        self._t = self._t + 1

                        # Update the weights + normalize
                        all_sum = 0.0
                        
                        for i in range(self._N):
                            all_sum = all_sum + self._n_weights[i]
                        
                        for i in range(self._N):
                            self._weights[i] = self._n_weights[i] / all_sum

                        # Update theta
                        for i in range(self._N):
                            self._theta[i] = self._n_theta[i]

                        # Update the covariance
                        self._cov = self.emp_var()
                    else:
                        self._i = self._i + 1

        # Otherwise try again                
        return None

        """
        fx = pints.vector(fx)
        accepted = self._xs[fx < self._threshold]
        if np.sum(accepted) == 0:
            return None
        else:
            return [self._xs.tolist() for c, x in
                    enumerate(accepted) if x] """

    def threshold(self):
        """
        Returns threshold error distance that determines if a sample is
        accepted (if ``error < threshold``).
        """
        return self._threshold

    def set_threshold(self, threshold):
        """
        Sets threshold error distance that determines if a sample is accepted
        (if ``error < threshold``).
        """
        x = float(threshold)
        if x <= 0:
            raise ValueError('Threshold must be greater than zero.')
        self._threshold = threshold
