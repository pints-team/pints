#
# Sequential Monte Carlo
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
# Some code in this file was adapted from Myokit (see http://myokit.org)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
from scipy import stats


class SMC(pints.SingleChainMCMC):
    """
    *Extends:* :class:`MCMC`
    Samples from a density using sequential Monte Carlo sampling [1].

    Algorithm 3.1.1 from:
    "Sequential Monte Carlo Samplers", Del Moral et al. 2006,
    Journal of the Royal Statistical Society. Series B.
    """
    def __init__(self, log_likelihood, x0, sigma0=None):
        super(SMC, self).__init__(
            log_likelihood, x0, sigma0)

        # Number of particles
        self._particles = 100

        # Total number of iterations
        self._iterations = self._dimension * 2000

        # Number of iterations to discard as burn-in
        self._burn_in = int(0.5 * self._iterations)

        # Thinning: Store only one sample per X
        self._thinning_rate = 1
    
        # Constant weights vector
        self._constant_weights = np.repeat(1.0 / num_samples, num_samples)
        
    def tempered_distribution(self,theta,beta):
        """
        Returns the log pdf of the tempered distribution 
        [p(theta|x) p(theta)]^(1-beta)
        """
        return (1.0 - beta) * self._log_likelihood(self)

    def burn_in(self):
        """
        Returns the number of iterations that will be discarded as burn-in in
        the next run.
        """
        return self._burn_in

    def iterations(self):
        """
        Returns the total number of iterations that will be performed in the
        next run, including the non-adaptive and burn-in iterations.
        """
        return self._iterations

    def run(self):

        # Report the current settings
        if self._verbose:
            print('Running sequential Monte Carlo')
            print('Total number of iterations: ' + str(self._iterations))
            print(
                'Number of iterations to discard as burn-in: '
                + str(self._burn_in))
            print('Storing 1 sample per ' + str(self._thinning_rate) + ' iteration')

        # Initial starting parameters
        mu = self._x0
        sigma = self._sigma0
        current = self._x0
        current_log_likelihood = self._log_likelihood(current)
        if not np.isfinite(current_log_likelihood):
            raise ValueError(
                'Suggested starting position has a non-finite log-likelihood.')

        # Starting parameters
        samples = np.random.multivariate_normal(mean=mu, cov=sigma,
                                                size=self._particles)
        
        # Starting weights
        Weights = np.zeros(self._particles)
        for i in range(0, self._particles):
          Weights[i] = np.exp(self.tempered_distribution(lSamples[i],1) - 
            stats.multivariate_normal.logpdf(Samples[i],mean=mu, cov=sigma))
        weights = weights / np.sum(weights)

        return weights

    def set_acceptance_rate(self, rate):
        """
        Sets the target acceptance rate for the next run.
        """
        rate = float(rate)
        if rate <= 0:
            raise ValueError('Target acceptance rate must be greater than 0.')
        elif rate > 1:
            raise ValueError('Target acceptance rate cannot exceed 1.')
        self._acceptance_target = rate

    def set_burn_in(self, burn_in):
        """
        Sets the number of iterations to discard as burn-in in the next run.
        """
        burn_in = int(burn_in)
        if burn_in < 0:
            raise ValueError('Burn-in rate cannot be negative.')
        self._burn_in = burn_in

    def set_iterations(self, iterations):
        """
        Sets the total number of iterations to be performed in the next run
        (including burn-in and non-adaptive iterations).
        """
        iterations = int(iterations)
        if iterations < 0:
            raise ValueError('Number of iterations cannot be negative.')
        self._iterations = iterations

    def set_non_adaptive_iterations(self, adaptation):
        """
        Sets the number of iterations to perform before using adapation in the
        next run.
        """
        adaptation = int(adaptation)
        if adaptation < 0:
            raise ValueError('Adaptation cannot be negative.')
        self._adaptation = adaptation

    def set_thinning_rate(self, thinning):
        """
        Sets the thinning rate. With a thinning rate of *n*, only every *n-th*
        sample will be stored.
        """
        thinning = int(thinning)
        if thinning < 1:
            raise ValueError('Thinning rate must be greater than zero.')
        self._thinning_rate = thinning

    def thinning_rate(self):
        """
        Returns the thinning rate that will be used in the next run. A thinning
        rate of *n* indicates that only every *n-th* sample will be stored.
        """
        return self._thinning_rate
    
    def tempered_distribution(self, x, beta):
        """
        Returns the tempered log-pdf:
        beta * log pi(x) + (1 - beta) * log N(0, sigma)
        """
        return beta * log_likelihood(x) + \
               (1.0 - beta) * stats.multivariate_normal.logpdf(x, mean=self._x0,
                                                               cov=self._sigma0)

    def resample(self, weights, samples):
        """
        Returns samples according to the weights vector
        from the multinomial distribution.
        """
        num_samples = len(weights)
        selected = np.random.multinomial(num_samples,weights)
        new_sample = np.zeros(0)
        for i in range(0,num_samples):
            if selected[i] > 0:
                new_sample = np.concatenate((new_sample,np.repeat(samples[i],selected[i])))
        return new_sample
    
    def w_tilde(x_old, x_new, beta_old, beta_new, sigma):
        numerator = tempered_distribution(x_new, beta_new) * L(x_new, x_old, beta_old, sigma)
        denominator = temperedDistribution(x_old, beta_old) * K(x_old, x_new, beta_new, sigma)
        return numerator / denominator

    def K(xOld,xNew,beta,aSigma):
        return stats.norm.pdf(xNew,loc=xOld,scale=aSigma) * temperedDistribution(xNew,beta) / temperedDistribution(xOld,beta)

    def L(xNew,xOld,beta,aSigma):
        return temperedDistribution(xOld,beta) * K(xOld,xNew,beta,aSigma) / temperedDistribution(xNew,beta)

    def newWeight(WOld,xOld,xNew,betaOld,betaNew,aSigma):
        wtilde = wTilde(xOld,xNew,betaOld,betaNew,aSigma)
        return WOld * wtilde

    # def newWeights(lWOld,lSamplesOld,lSamplesNew,betaOld,betaNew,aSigma):
    #     lNewW = map(lambda (WOld,xOld,xNew): newWeight(WOld,xOld,xNew,betaOld,betaNew,aSigma),zip(lWOld,lSamplesOld,lSamplesNew))
    #     return lNewW / np.sum(lNewW)
    # 
    # def kernel_sample(samples, beta, sigma):
    #     proposed = map(lambda x: np.random.multivariate_normal(mean=x, cov=sigma), samples)
    #     new_samples = map(lambda (old, new): old if (tempered_distribution(new, beta) /
    #                                                  tempered_Distribution(old, beta))
    #                                                  <= np.random.uniform(size=1) /
    #                                              else new,
    #                       zip(samples, proposed))
    #     return new_samples
    # 
    # def steps2And3(samples_old, weights_old, beta_old, beta_new, sigma):
    #     """
    #     Undertakes steps 2 and 3 from algorithm 3.1.1. in
    #     Del Moral 2006 paper.
    #     """
    #     num_samples = len(weights_old)
    #     resamples = resample(weights_old, samples_old)
    #     samples_new = kernel_sample(resamples, beta_new, sigma)
    #     weights_new = new_weights(self._constant_weights,
    #                               samples_old, samples_new,
    #                               beta_old, beta_new, sigma)
    #     resamples_new, weights_new = resample(weights_new, samples_new)
    #     return lResamplesNew

