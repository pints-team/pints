#
# Population Monte Carlo Approximate Bayesian Computation
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import numpy as np
from scipy.stats import multivariate_normal


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
    def __init__(self, log_prior, eps_ratio=0.99):
        self._log_prior = log_prior
        self._xs = None
        self._ready_for_tell = False
        self._weights = np.array([])
        self._eps = 1
        self._T = 10
        self._t = 1
        self._i = 0
        self._eps_ratio = eps_ratio

    def name(self):
        """ See :meth:`pints.ABCSampler.name()`. """
        return 'PMC ABC'

    def emp_var(self):
        """ Computes the weighted empirical variance of self._theta. """
        # Compute weighted mean
        w_mean = np.zeros(self._dim)
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
        partial_mat = np.zeros((self._dim, self._dim))
        for i in range(self._N):
            diff = self._theta[i] - w_mean
            for j in range(self._dim):
                for k in range(self._dim):
                    partial_mat[j][k] = diff[j] * diff[k]
            n_V = n_V + self._weights[i] * partial_mat
        
        # Add correction term
        if w_sum ** 2 == w_sq_sum:
            e_var = (w_sum ** 2) / 1e-20 * n_V
        else:
            e_var = ((w_sum ** 2) / ((w_sum ** 2) - w_sq_sum)) * n_V

        return e_var

    def ask(self, n_samples):
        """ See :meth:`ABCSampler.ask()`. """
        if self._ready_for_tell:
            raise RuntimeError('Ask called before tell.')

        self._ready_for_tell = True

        if self._t == 1:
            if self._i == 0:
                # Initialize variables dependent on N
                self._dim = self._log_prior.n_parameters()
                self._N = n_samples
                self._i = 1
                self._theta = np.zeros((self._N + 1, self._dim))
                self._n_theta = np.zeros((self._N + 1, self._dim))
                self._xs = self._log_prior.sample(self._N)
                self._weights = np.zeros(self._N + 1)
                self._n_weights = np.zeros(self._N + 1)
                for i in range(self._N):
                    self._weights[i] = 1.0 / self._N

            # Sample theta_i
            self._xs = self._log_prior.sample(1)
        else:
            done = False
            while not done:
                # Sample theta_star
                pt = np.random.uniform()
                uninitialized = True
                theta_star = np.zeros(self._dim)
                partial_sum = 0
                for i in range(self._N):
                    if uninitialized and pt <= partial_sum + self._weights[i]:
                        theta_star = self._theta[i]
                        uninitialized = False
                    else:
                        partial_sum = partial_sum + self._weights[i]

                # Generate sample
                if self._dim == 1:
                    self._n_theta[i] = [np.random.normal(theta_star,
                                                         self._cov)]
                else:
                    self._n_theta[i] = np.random.multivariate_normal(
                        mean=theta_star, cov=self._cov)

                self._xs = [self._n_theta[i]]

                # Assure that the value is within the prior
                if self._log_prior(self._xs) != np.NINF:
                    done = True

        return self._xs

    def tell(self, fx):
        """ See :meth:`ABCSampler.tell()`. """
        if not self._ready_for_tell:
            raise RuntimeError('Tell called before ask.')
        self._ready_for_tell = False

        if len(fx) != 1:
            raise RuntimeError('Expected only 1 error term.')

        if self._t == 1:
            if fx[0] < self._eps:
                # Write the definite value of theta_i^1
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
            if fx[0] < self._eps:
                self._n_theta[self._i] = self._xs[0]
                if self._i == self._N and self._t == self._T:
                    # Finished
                    return self._n_theta
                else:
                    # Update weight i
                    norm_term = 0.0
                    for j in range(self._N):
                        norm_term = norm_term + self._weights[j] * \
                            multivariate_normal(self._n_theta[self._i],
                                                self._cov).pdf(self._theta[j])

                    # Preventing numerical errors
                    if norm_term == 0.0:
                        norm_term = 1e-20

                    self._n_weights[self._i] = (self._log_prior(
                                                self._n_theta[self._i])
                                                / norm_term)
                    if self._i == self._N:
                        # Update epsilon
                        self._eps = self._eps * self._eps_ratio
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

    def threshold(self):
        """
        Returns threshold error distance that determines if a sample is
        accepted (if ``error < threshold``).
        """
        return self._eps

    def set_threshold(self, threshold):
        """
        Sets threshold error distance that determines if a sample is accepted
        (if ``error < threshold``).
        """
        x = float(threshold)
        if x <= 0:
            raise ValueError('Threshold must be greater than zero.')
        self._eps = threshold

    def set_n_generations(self, n_gen):
        """
        Sets the number of generations used in PMC, called T in the original
        paper.
        """
        x = int(n_gen)
        if x <= 0:
            raise ValueError('Number of generations must be greater than' +
                             'zero.')
        self._T = x

    def set_t_ratio(self, t_ratio):
        """
        Sets the rate by which the threshold is multiplied after each
        generation, so that each generation gets a more accurate tighter
        threshold and more accurate iterations.
        """
        x = float(t_ratio)
        if x < 0.0 or x > 1.0:
            raise ValueError('Threshold ration must be between 0.0 and 1.0')

        self._eps_ratio = x
