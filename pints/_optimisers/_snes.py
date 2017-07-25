#
# Seperable natural evolution strategy optimizer: SNES
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
# Some code in this file was adapted from Myokit (see http://myokit.org)
#
import pints
import numpy as np
import multiprocessing

class SNES(pints.Optimiser):
    """
    Finds the best parameters using the SNES method described in [1, 2].
    
    SNES stands for Seperable Natural Evolution Strategy, and is designed for
    non-linear derivative-free optimization problems in high dimensions and
    with many local minima [1].
    
    It treats each dimension separately, making it suitable for higher
    dimensions.
         
    [1] Schaul, Glasmachers, Schmidhuber (2011) High dimensions and heavy tails
    for natural evolution strategies.
    Proceedings of the 13th annual conference on Genetic and evolutionary
    computation. ACM, 2011.
    
    [2] PyBrain: The Python machine learning library (http://pybrain.org)
   
    """
    def run(self):

        # Default search parameters
        parallel = True
        
        # Search is terminated after max_iter iterations
        max_iter = 10000

        # Or if the result doesn't change significantly for a while
        max_unchanged_iterations = 100
        min_significant_change = 1e-11
        unchanged_iterations = 0

        # Parameter space dimension
        d = self._dimension

        # Population size
        # If parallel, round up to a multiple of the reported number of cores
        n = 4 + int(3 * np.log(d))
        if parallel:            
            cpu_count = multiprocessing.cpu_count()
            n = (((n - 1) // cpu_count) + 1) * cpu_count

        # Set up progress reporting in verbose mode
        nextMessage = 0
        if self._verbose:
            if parallel:
                print('Running in parallel mode with population size '
                    + str(n))
            else:
                print('Running in sequential mode with population size '
                    + str(n))

        # Apply wrapper to implement boundaries
        if self._boundaries is None:
            xtransform = lambda x: x
        else:
            xtransform = pints.TriangleWaveTransform(self._boundaries)

        # Create evaluator object
        if parallel:
            evaluator = pints.ParallelEvaluator(self._function)
        else:
            evaluator = pints.SequentialEvaluator(self._function)

        # Learning rates
        eta_mu = 1
        eta_sigmas = 0.2 * (3 + np.log(d)) * d ** -0.5

        # Pre-calculated utilities
        us = np.maximum(0, np.log(n / 2 + 1) - np.log(1 + np.arange(n)))
        us /= np.sum(us)
        us -= 1/n

        # Center of distribution
        mu = self._x0

        # Initial square root of covariance matrix
        sigmas = np.array(self._sigma0, copy=True)
        
        # Best solution found
        xbest = mu
        fbest = float('inf')

        # Start running
        for iteration in xrange(1, 1 + max_iter):
        
            # Create new samples
            ss = np.array([np.random.normal(0, 1, d) for i in xrange(n)])
            xs = mu + sigmas * ss
            
            # Evaluate at the samples
            fxs = evaluator.evaluate(xtransform(xs))
            
            # Order the normalized samples according to the scores
            order = np.argsort(fxs)
            ss = ss[order]
            
            # Update center
            mu += eta_mu * sigmas * np.dot(us, ss)

            # Update variances
            sigmas *= np.exp(0.5 * eta_sigmas * np.dot(us, ss**2 - 1))
            
            # Update best if needed
            if fxs[order[0]] < fbest:
                
                # Check if this counts as a significant change
                fnew = fxs[order[0]]
                if np.sum(np.abs(fnew - fbest)) < min_significant_change:
                    unchanged_iterations += 1
                else:
                    unchanged_iterations = 0
            
                # Update best
                xbest = xs[order[0]]
                fbest = fnew
                
            else:
                unchanged_iterations += 1
            
            # Show progress in verbose mode:
            if self._verbose and iteration >= nextMessage:
                print(str(iteration) + ': ' + str(fbest))
                if iteration < 3:
                    nextMessage = iteration + 1
                else:
                    nextMessage = 20 * (1 + iteration // 20)
            
            # Stop if no change for too long
            if unchanged_iterations >= max_unchanged_iterations:
                if self._verbose:
                    print('Halting: No significant change for '
                        + str(unchanged_iterations) + ' iterations.')
                break
            
        # Show stopping criterion
        if self._verbose and unchanged_iterations < max_unchanged_iterations:
            print('Halting: Maximum iterations reached.')
        
        # Get final score at mu
        fmu = self._function(xtransform(mu))
        if fmu < fbest:
            if self._verbose:
                print('Final score at mu beats best sample')
            xbest = mu
            fbest = fmu
        
        # Show final value
        if self._verbose:
            print(str(iteration) + ': ' + str(fbest))
        
        # Return best solution
        return xtransform(xbest), fbest

def snes(function, boundaries=None, x0=None, sigma0=None):
    """
    Runs a SNES optimisation with the default settings.
    """
    return SNES(function, boundaries, x0, sigma0).run() 

