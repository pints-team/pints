#
# Exponential natural evolution strategy optimizer: xNES
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import numpy as np
import scipy
import scipy.linalg
import multiprocessing

class XNES(pints.Optimiser):
    """
    Finds the best parameters using the xNES method described in [1, 2].
    
    xNES stands for Exponential Natural Evolution Strategy, and is
    designed for non-linear derivative-free optimization problems [1].
        
    [1] Glasmachers, Schaul, Schmidhuber et al. (2010) Exponential natural
    evolution strategies.
    Proceedings of the 12th annual conference on Genetic and evolutionary
    computation
    
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
        eta_A = 0.6 * (3 + np.log(d)) * d ** -1.5

        # Pre-calculated utilities
        us = np.maximum(0, np.log(n / 2 + 1) - np.log(1 + np.arange(n)))
        us /= np.sum(us)
        us -= 1/n

        # Center of distribution
        mu = self._hint

        # Guess initial square root of covariance matrix
        A = np.eye(d)
        
        # Improve guess of covariance using boundaries or hint
        if self._boundaries:
            A *= (self._boundaries._upper - self._boundaries._lower) / 6.0
        else:
            # Use hint to guess at parameter scaling
            # If no hint is given, a hint may have been set based on user given
            # boundaries, if they're not given, it'll simply be zero
            if np.sum(np.abs(self._hint)) > 1e-11: # not zero
                A *= np.abs(self._hint) / np.sum(np.abs(self._hint))
        
        # Show initial guess of square of covariance
        if self._verbose:
            print('Initial guess for square of covariance matrix:')
            print(A)
        
        # Identity matrix for later use
        I = np.eye(d)

        # Best solution found
        xbest = mu
        fbest = float('inf')

        # Start running
        for iteration in xrange(1, 1 + max_iter):
        
            # Create new samples
            zs = np.array([np.random.normal(0, 1, d) for i in xrange(n)])
            xs = np.array([mu + np.dot(A, zs[i]) for i in xrange(n)])
            
            # Evaluate at the samples
            fxs = evaluator.evaluate(xtransform(xs))
            
            # Order the normalized samples according to the scores
            order = np.argsort(fxs)
            zs = zs[order]
            
            # Update center
            Gd = np.dot(us, zs)
            mu += eta_mu * np.dot(A,  Gd)
            
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
            
            # Update root of covariance matrix
            Gm = np.dot(np.array([np.outer(z, z).T - I for z in zs]).T, us)
            A *= scipy.linalg.expm(np.dot(0.5 * eta_A, Gm))
            
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

def xnes(function, boundaries=None, hint=None):
    """
    Runs an XNES optimisation with the default settings.
    """
    return XNES(function, boundaries, hint).run() 

