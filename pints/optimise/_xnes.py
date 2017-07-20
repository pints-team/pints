#
# Exponential natural evolution strategy optimizer: xNES
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import pints.optimise
import scipy
import scipy.linalg

class XNES(pints.optimise.Optimiser):
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
    def __init__(model, times, values, lower, upper, hint=None):
        super(XNES, self).__init__(model, times, values, lower, upper, hint)

    def fit(self, parallel=True, verbose=True):

        # Check if parallelization is required
        parallel = bool(parallel)

        # Check if in verbose mode
        verbose = bool(verbose)
        
        # Create periodic parameter space transform to implement boundaries
        transform = pints.optimise._TriangleWaveTransform(self._lower,
            self._upper)
    
        # Wrap transform around score function
        function = pints.optimise._ParameterTransformWrapper(f, transform)

        # Default search parameters
        # Search is terminated after max_iter iterations
        max_iter = 10000
        # Or if ntol iterations don't improve the best solution by more than
        # ftol
        ntol = 20
        ftol = 1e-11

        # Parameter space dimension
        d = self._model.dimension()

        # Population size
        n = 4 + int(3 * np.log(d))
        # If parallel, round up to a multiple of the reported number of cores
        if parallel:
            cpu_count = multiprocessing.cpu_count()
            n = (((n - 1) // cpu_count) + 1) * cpu_count

        # Set up progress reporting in verbose mode
        nextMessage = 0
        if verbose:
            if parallel:
                print('Running in parallel mode with population size '
                    + str(n))
            else:
                print('Running in sequential mode with population size '
                    + str(n))

        # Create evaluator object
        if parallel:
            evaluator = ParallelEvaluator(function, args=args)
        else:
            evaluator = SequentialEvaluator(function, args=args)

        # Set up algorithm

        # Learning rates
        eta_mu = 1
        eta_A = 0.6 * (3 + np.log(d)) * d ** -1.5

        # Pre-calculated utilities
        us = np.maximum(0, np.log(n / 2 + 1) - np.log(1 + np.arange(n)))
        us /= np.sum(us)
        us -= 1/n

        # Center of distribution
        mu = self._hint

        # Square root of covariance matrix
        A = np.eye(d)

        # Identity matrix for later use
        I = np.eye(d)

        # Best solution found
        xbest = hint
        fbest = function(hint, *args)

        # Report first point
        if callback is not None:
            callback(np.array(hint, copy=True), fbest)

        # Start running
        for iteration in xrange(1, 1 + max_iter):
            # Create new samples
            zs = np.array([np.random.normal(0, 1, d) for i in xrange(n)])
            xs = np.array([mu + np.dot(A, zs[i]) for i in xrange(n)])
            # Evaluate at the samples
            fxs = evaluator.evaluate(xs)
            # Order the normalized samples according to the scores
            order = np.argsort(fxs)
            zs = zs[order]
            # Update center
            Gd = np.dot(us, zs)
            mu += eta_mu * np.dot(A,  Gd)
            # Update best if needed
            if fxs[order[0]] < fbest:
                xbest = xs[order[0]]
                fbest = fxs[order[0]]
                # Target reached? Then break and return
                if fbest <= target:
                    # Report to callback if requested
                    if callback is not None:
                        callback(transform(xbest), fbest)
                    if verbose:
                        print('Target reached, halting')
                    break
            # Report to callback if requested
            if callback is not None:
                callback(transform(xbest), fbest)
            # Show progress in verbose mode:
            if verbose:
                if iteration >= nextMessage:
                    print(str(iteration) + ': ' + str(fbest))
                    if iteration < 3:
                        nextMessage = iteration + 1
                    else:
                        nextMessage = 100 * (1 + iteration // 100)
            # Update root of covariance matrix
            Gm = np.dot(np.array([np.outer(z, z).T - I for z in zs]).T, us)
            A *= scipy.linalg.expm(np.dot(0.5 * eta_A, Gm))
        # Show stopping criterion
        if fbest > target:
            if verbose:
                print('Maximum iterations reached, halting')
        # Get final score at mu
        fmu = function(mu, *args)
        if fmu < fbest:
            if verbose:
                print('Final score at mu beats best sample')
            xbest = mu
            fbest = fmu
        # Show final value and return
        if verbose:
            print(str(iteration) + ': ' + str(fbest))
        return transform(xbest), fbest

