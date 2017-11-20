#
# Exponential natural evolution strategy optimizer: xNES
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
# Some code in this file was adapted from Myokit (see http://myokit.org)
#
import pints
import os
import numpy as np
import scipy
import scipy.linalg
import multiprocessing
import time

class AdaptiveCovarianceMCMC(pints.MCMC):
    """
    Creates a chain of samples from a target distribution, using the adaptive
    covariance routine described in [1].
        
    [1] Uncertainty and variability in models of the cardiac action potential:
    Can we build trustworthy models?
    Johnstone, Chang, Bardenet, de Boer, Gavaghan, Pathmanathan, Clayton,
    Mirams (2015) Journal of Molecular and Cellular Cardiology
    """

    def set_default(self):
        """
        Set MCMC routine required variables as default.

        # Target acceptance rate
        acceptance_target = 0.25
        # Total number of iterations
        # Default set for small iteration (linear grow)
        iterations = 2000 * dimension
        # Number of iterations before using adapation
        adaptation = 0.25 * iterations
        # Number of iterations to discard as burn-in
        # Only make sense for small iterations
        burn_in = 0.75 * iterations
        # Thinning: Store only one sample per X
        thinning = 4
        """
        # Target acceptance rate
        self._acceptance_target = 0.25
        # Total number of iterations
        self._iterations = self._dimension * 2000
        # Number of iterations before adapation
        self._adaptation_rate = 0.25
        self._adaptation = int(self._iterations * self._adaptation_rate)
        # Number of iterations to discard as burn-in
        self._burn_in_rate = 0.75
        self._burn_in = int(self._iterations * self._burn_in_rate)
        # Thinning: Store only one sample per X
        self._thinning = 4
    
    def acceptance_rate(self):
        """
        Return target acceptance rate.
        """
        return self._acceptance_target

    def iterations(self):
        """
        Return the total number of iterations.
        """
        return self._iterations

    def adaptation(self):
        """
        Return the number of iterations before adapation.
        """
        return self._adaptation

    def adaptation_rate(self):
        """
        Return the first fraction of the total number of iterations before
        using adaptation.
        """
        return self._adaptation_rate

    def burn_in(self):
        """
        Return the number of iterations to discard as burn-in.
        """
        return self._burn_in

    def burn_in_rate(self):
        """
        Return the first fraction of the total number of iterations to discard as
        burn-in.
        """
        return self._burn_in_rate

    def thinning(self):
        """
        Return thinning.
        """
        return self._thinning

    def set_acceptance_rate(self, rate):
        """
        Set target acceptance rate.
        """
        self._acceptance_target = rate

    def set_iterations(self, iterations):
        """
        Set the total number of iterations.
        """
        self._iterations = iterations
        self._adaptation = int(self._iterations * self._adaptation_rate)
        self._burn_in = int(self._iterations * self._burn_in_rate)
        if self._thinning > self._adaptation:
            print('WARNING: Thinning larger than adaptation')
        if self._thinning > self._burn_in:
            print('WARNING: Thinning larger than burn-in')
        if self._thinning > self._iterations:
            raise Exception('Thinning larger than total number of iterations')

    def set_thinning(self, thinning):
        """
        Set thinning - store only one sample per X
        """
        self._thinning = thinning
        if self._thinning > self._adaptation:
            print('WARNING: Thinning larger than adaptation')
        if self._thinning > self._burn_in:
            print('WARNING: Thinning larger than burn-in')
        if self._thinning > self._iterations:
            raise Exception('Thinning larger than total number of iterations')

    def set_adaptation_rate(self, rate):
        """
        Set the first fraction of the total number of iterations before 
        using adaptation.
        """
        self._adaptation_rate = rate
        self._adaptation = int(self._iterations * self._adaptation_rate)
        if self._adaptation > self._burn_in:
            print('WARNING: Adaptation use after burn-in')

    def set_adaptation(self, adaptation):
        """
        Set the number of iterations before using adapation.
        """
        self._adaptation = adaptation
        self._adaptation_rate = self._adaptation/self._iterations
        if self._adaptation > self._burn_in:
            print('WARNING: Adaptation use after burn-in')

    def set_burn_in_rate(self, rate):
        """
        Set the first fraction of the total number of iterations to discard as 
        burn-in.
        """
        self._burn_in_rate = rate
        self._burn_in = int(self._iterations * self._burn_in_rate)
        if self._adaptation > self._burn_in:
            print('WARNING: Adaptation use after burn-in')

    def set_burn_in(self, burn_in):
        """
        Set the number of iterations to discard as burn-in.
        """
        self._burn_in = burn_in
        self._burn_in_rate = self._burn_in/self._iterations
        if self._adaptation > self._burn_in:
            print('WARNING: Adaptation use after burn-in')

    def print_setup(self):
        """
        Print to console the current setup info of the adaptive convariance
        MCMC routine.
        """
        print('\n## Adaptive convariance MCMC routine setup info')
        print('Target acceptance rate: ' + str(self._acceptance_target))
        print('Total number of iterations: ' + str(self._iterations))
        print('Number of iterations before adapation: ' + str(self._adaptation))
        print('Number of iterations to discard as burn-in: ' + str(self._burn_in))
        print('Thinning: Store only one sample per ' + str(self._thinning) + '\n')

    def run(self):
        """
        Run adaptive convariance MCMC routine and return a chain of samples 
        from the target distribution.
        """

        # Time it
        start = time.time()

        # Problem dimension
        d = self._dimension

        # Target acceptance rate
        acceptance_target = self._acceptance_target

        # Total number of iterations
        iterations = self._iterations

        # Number of iterations before adapation
        adaptation = self._adaptation

        # Number of iterations to discard as burn-in
        burn_in = self._burn_in

        # Thinning: Store only one sample per X
        thinning = self._thinning

        # Print out what we are using once
        self.print_setup()

        # Initial starting parameters
        mu = self._x0
        sigma = self._sigma0
        current = self._x0
        current_log_likelihood = self._log_likelihood(current)
        if not np.isfinite(current_log_likelihood):
            raise ValueError('Suggested starting position has a non-finite'
                ' log-likelihood')

        # Chain of stored samples
        stored = int((iterations - burn_in) / thinning)
        self._chain = np.zeros((stored, d))
        chain = self._chain

        # Initial acceptance rate (value doesn't matter)
        loga = 0
        acceptance = 0
        
        # Save to file
        if self._save:
            savedir = os.path.join(self._save_dir, 'adaptive-mcmc.txt')
            with open(savedir, 'w') as outfile:
                outfile.write('# Adaptive covariance matrix\n')
                outfile.write('# Thinned by taking saving only every ' 
                              + str(thinning) + '-th accepted state\n')
                outfile.write('# 1-'+str(d)+'. parameters, '
                              +str(1+d)+'. logLikelihood, '
                              +str(2+d)+'. acceptance rate, '
                              +str(3+d)+'. loga, '
                              +str(4+1*d)+'-'+str(4+2*d-1)+'. mean estimate, '
                              +str(4+2*d)+'. real time taken (s), '
                              +str(5+2*d)+'. iteration\n')
            H = [np.concatenate((np.copy(current), 
                                [current_log_likelihood, acceptance, loga], 
                                mu, 
                                [time.time()-start,0]
                                ))]

        # Go!
        for i in xrange(iterations):
            # Propose new point
            # Note: Normal distribution is symmetric
            #  N(x|y, sigma) = N(y|x, sigma) so that we can drop the proposal
            #  distribution term from the acceptance criterion
            proposed = np.random.multivariate_normal(current,
                np.exp(loga) * sigma)
            
            # Check if the point can be accepted
            accepted = 0
            proposed_log_likelihood = self._log_likelihood(proposed)
            if np.isfinite(proposed_log_likelihood):
                u = np.log(np.random.rand())
                if u < proposed_log_likelihood - current_log_likelihood:
                    accepted = 1
                    current = proposed
                    current_log_likelihood = proposed_log_likelihood
            
            # Adapt covariance matrix
            if i >= adaptation:
                gamma = (i - adaptation + 1) ** -0.6
                dsigm = np.reshape(current - mu, (d, 1))
                sigma = (1 - gamma) * sigma + gamma * np.dot(dsigm, dsigm.T)
                mu = (1 - gamma) * mu + gamma * current
                loga += gamma * (accepted - acceptance_target)

            # Update acceptance rate
            acceptance = (i * acceptance + float(accepted)) / (i + 1)
            
            # Add point to chain
            ilog = i - burn_in
            if ilog >= 0 and ilog % thinning == 0:
                chain[ilog // thinning, :] = current
                if self._save:
                    # Only save after burn-in
                    H.append(np.concatenate((
                        np.copy(current), 
                        [current_log_likelihood, acceptance, loga], 
                        mu, 
                        [time.time()-start,i]
                        )))
            
            # Report
            if self._verbose and i % 50 == 0:
                print('Iteration ' + str(i) + ' of ' + str(iterations))
                print('  In burn-in: ' + str(i >= burn_in))
                print('  Adapting: ' + str(i >= adaptation))
                print('  Acceptance rate: ' + str(acceptance))
            # Save
            if self._save and i % 500 == 0:
                with open(savedir, 'a') as outfile:
                    np.savetxt(outfile, H) 
                H = []

        # Check that chain fully filled
        if ilog // thinning != len(chain) - 1:
            raise Exception('Unexpected error: Chain not fully generated.')

        # Return generated chain
        return chain


    def load_chain(self, path_or_chain):
        """
        Load previously simulated chain for later anaylsis for
        adpative MCMC method.

        Arguments:

        ``path_or_chain``
            String: Path to saved chain.
            Array-type: Load it as current chain

        """
        d = self._dimension
        if isinstance(path_or_chain, str):
            try:
                iterations, final_loga, time_taken = np.loadtxt(path_or_chain, 
                        usecols=[5+2*d-1, 3+d-1, 4+2*d-1])[-1,:]
                iterations_2 = np.loadtxt(path_or_chain,
                        usecols=[5+2*d-1])[-2]
                iterations_burn_in = np.loadtxt(path_or_chain,
                        usecols=[5+2*d-1])[1]
                self.set_iterations(int(iterations))
                self.set_thinning(int(iterations - iterations_2))
                self.set_burn_in(int(iterations_burn_in))
                self._chain = np.loadtxt(path_or_chain, 
                        usecols=range(d))[1:,:]
            except:
                raise Exception('Cannot load file as chain ' 
                                + path_or_chain)
        elif isinstance(path_or_chain, (np.ndarray, list)):
            try:
                assert path_or_chain.shape[1] == d
                self._chain = path_or_chain
            except:
                raise Exception('Array does not match expected chain dimension')

    def chain(self):
        """
        Return chain.
        """
        return self._chain


def adaptive_covariance_mcmc(log_likelihood, x0, sigma0=None, savetofile=False):
    """
    Runs an adaptive covariance MCMC routine with the default parameters.
    """
    return AdaptiveCovarianceMCMC(log_likelihood, x0, sigma0, savetofile=savetofile).run() 

