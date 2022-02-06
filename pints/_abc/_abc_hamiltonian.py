#
# Hamiltonian ABC method
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import numpy as np
from scipy.stats import multivariate_normal


class SyntheticLikelihood:
    def __init__(self, y, eps):
        y = np.array(y)
        if len(y.shape) == 1:
            self._y = np.array([y])
        else:
            self._y = np.array(y)
        self._eps = eps

    def pdf(self, vals):
        the_sum = 0.0
        for i in range(len(vals)):
            final_res = 1.0
            for j in range(len(self._y)):
                term = multivariate_normal.pdf(self._y[j], mean=vals[i][j], cov=(self._eps * np.transpose(self._eps)) )
                final_res = final_res + term
                
            the_sum = the_sum + final_res
        
        if the_sum == 0:
            the_sum = np.NINF
        else:
            the_sum = np.log(the_sum)
        
        return the_sum



# try initially with fdsa
class HamiltonianABC(pints.ABCSampler):
    r"""
    Implements the Hamiltonian ABC algorithm as described in [1].

    Here is a high-level description of the algorithm:

    .. math::
        \begin{align}
        \theta^* &\sim p(\theta) \\
        x &\sim p(x|\theta^*) \\
        \textrm{if } s(x) < \textrm{threshold}, \textrm{then} \\
        \theta^* \textrm{ is added to list of samples} \\
        \end{align}

    In other words, the first two steps sample parameters
    from the prior distribution :math:`p(\theta)` and then sample
    simulated data from the sampling distribution (conditional on
    the sampled parameter values), :math:`p(x|\theta^*)`.
    In the end, if the error measure between our simulated data and
    the original data is within the threshold, we add the sampled
    parameters to the list of samples.

    References
    ----------
    .. [1] "Approximate Bayesian Computation (ABC) in practice". Katalin
           Csillery, Michael G.B. Blum, Oscar E. Gaggiotti, Olivier Francois
           (2010) Trends in Ecology & Evolution
           https://doi.org/10.1016/j.tree.2010.04.001

    """
    def __init__(self, log_prior, y, c=0.01, m=5, eps=0.01):
        self._m = m
        self._c = c
        self._cnt = 0
        
        # Functions
        self._log_prior = log_prior
        self._grad_prior = self.grad_pr 
   
    def name(self):
        """ See :meth:`pints.ABCSampler.name()`. """
        return 'Hamiltonian ABC'

    def grad_pr(self, theta):
        x, dx = self._log_prior.evaluateS1(theta)
        return dx

    def adapt_cov(self):
        self._mean = np.zeros(len(self._grads[0]))
        N = len(self._grads)
        
        for i in range(N):
            self._mean = (self._mean * i + self._grads[i]) / (i + 1)
        
        
        sum_term = np.zeros((self._dim, self._dim))
        
        for i in range(N):
            sum_term = sum_term + ( np.array( self._grads[i] - self._mean ) * np.transpose( self._grads[i] - self._mean ) )
        
        if N < 2:
            self._cov_matrix = np.eye(self._dim)
        else:
            self._cov_matrix = (1 / (N - 1)) * sum_term
        
        # Update B, C
        self._B = (self._eps / 2) * self._cov_matrix
        self._C = self._c * np.eye(self._dim) + self._cov_matrix
    
    def spsa(self, theta, d_theta, S, R):
        g = np.zeros(len(theta))
        self._grads = None
        
        for r in range(R):
            # Generate bernoulli distribution vector
            delta = np.zeros(len(theta))
            for i in range(len(theta)):
                delta[i] = (2 * np.random.binomial(n=1, p=0.5) - 1)
            
            set_plus = None
            set_minus = None
            
            for s in range(S):
                x_plus = self._sim_f(theta + d_theta * delta)
                x_minus = self._sim_f(theta - d_theta * delta)
                
                
                if len(x_plus.shape) == 1:
                    x_plus = [[x] for x in x_plus]
                if len(x_minus.shape) == 1:
                    x_minus = [[x] for x in x_minus]
                
                difference = ( ( self._synt_l.pdf([x_plus]) - self._synt_l.pdf([x_minus]) ) / (2 * d_theta) ) * delta
                if self._grads is None:
                    self._grads = [difference]
                else:
                    self._grads.append(difference)
                
                if set_plus is None:
                    set_plus = [x_plus]
                    set_minus = [x_minus]
                else:
                    set_plus.append(x_plus)
                    set_minus.append(x_minus)
            
            aux = (self._synt_l.pdf(set_plus) - self._synt_l.pdf(set_minus)) * delta
            g = g + aux
            
        self.adapt_cov()
        g = g / (2 * d_theta * R)
        
        grad_val = self.grad_pr(theta)
        if len(grad_val.shape) > 1:
            grad_val = grad_val[0]
        g += grad_val
        
        return -g
    

    def ask(self, n_samples):
        """ See :meth:`ABCSampler.ask()`. """
        if self._ready_for_tell:
            raise RuntimeError('Ask called before tell.')
        
        if self._i == 0:
            self._returnable = [self._results[0]]
        else:
            self._returnable = []
        for t in range(self._i, n_samples):
            # Resample momentum
            curr_theta = self._results[t-1]
            curr_momentum = np.random.multivariate_normal(np.zeros(self._dim), np.eye(self._dim))
            for i in range(self._m):
                next_theta = curr_theta + self._eps * curr_momentum
                # We need to process the fdsa term to update the matrices
                spsa_term = self.spsa(next_theta, 0.001, S=4, R=4)
                next_momentum = curr_momentum - self._eps * spsa_term \
                                - self._eps * self._C * curr_momentum \
                                + np.random.multivariate_normal(np.zeros(self._dim), 2 * self._eps * (self._C - self._B))
                curr_theta = next_theta
                curr_momentum = next_momentum

            self._results[t] = curr_theta
            self._returnable.append(curr_theta)
        
        n_samples += self._returnable

        self._ready_for_tell = True
        return np.array([])

    def tell(self, fx):
        """ See :meth:`ABCSampler.tell()`. """
        if not self._ready_for_tell:
            raise RuntimeError('Tell called before ask.')
        self._ready_for_tell = False
        
        return self._returnable

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
        self._eps = x

        # Build synthetic likelihood
        self._synt_l = SyntheticLikelihood(y, self._eps)

    def set_theta0(self, theta0):
        """
        Sets the first element in the chain of thetas.
        """
        self._theta0 = theta0
        self._dim = len(theta0)
        self._mean = np.zeros(self._dim)
        
        # Matrices
        self._cov_matrix = np.eye(self._dim)
        self._C = (self._c + 1) * np.eye(self._dim)
        self._B = (self._eps / 2) * np.eye(self._dim)
        self._grads = []
        
        self._results = np.zeros((self._T, self._dim))
        self._results[0] = self._theta0

    def set_sim_f(self, sim_f):
        """
        Set simulating function.
        """
        self._sim_f = sim_f