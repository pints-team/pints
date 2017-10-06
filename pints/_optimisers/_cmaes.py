#
# Uses the Python `cma` module to runs CMA-ES optimisations.
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
# Some code in this file was adapted from Myokit (see http://myokit.org)
#
import pints
import multiprocessing
import numpy as np

class CMAES(pints.Optimiser):
    """
    Finds the best parameters using the CMA-ES method described in [1, 2] and
    implemented in the `cma` module.

    CMA-ES stands for Covariance Matrix Adaptation Evolution Strategy, and is
    designed for non-linear derivative-free optimization problems.

    [1] https://www.lri.fr/~hansen/cmaesintro.html

    [2] Hansen, Mueller, Koumoutsakos (2006) Reducing the time complexity of
    the derandomized evolution strategy with covariance matrix adaptation
    (CMA-ES).

    """
    def run(self):

        # Import cma (may fail!)
        # Only the first time this is called in a running program incurs
        # much overhead.
        import cma

        # Default search parameters
        #TODO Allow changing before run() with method call
        parallel = True

        # Number of IPOP repeats (0=no repeats, 1=repeat once=2 runs, etc.)
        #TODO Allow changing before run() with method call
        ipop = 0

        # Parameter space dimension
        d = self._dimension

        # Population size
        #TODO Allow changing before run() with method call
        # If parallel, round up to a multiple of the reported number of cores
        # In IPOP-CMAES, this will be used as the _initial_ population size
        n = 4 + int(3 * np.log(d))
        if parallel:
            cpu_count = multiprocessing.cpu_count()
            n = (((n - 1) // cpu_count) + 1) * cpu_count

        # Search is terminated after max_iter iterations
        #TODO Allow changing before run() with method call
        max_iter = 10000
        # CMA-ES default: 100 + 50 * (d + 3)**2 // n**0.5

        # Or if successive iterations do not produce a significant change
        #TODO Allow changing before run() with method call
        #max_unchanged_iterations = 100
        min_significant_change = 1e-11
        #unchanged_iterations = 0
        # CMA-ES max_unchanged_iterations fixed value: 10 + 30 * d / n

        # Create evaluator object
        if parallel:
            evaluator = pints.ParallelEvaluator(self._function)
        else:
            evaluator = pints.SequentialEvaluator(self._function)

        # Set up simulation
        options = cma.CMAOptions()

        # Set boundaries
        if self._boundaries is not None:
            options.set('bounds',
                [list(self._boundaries._lower), list(self._boundaries._upper)])

        # Set stopping criteria
        options.set('maxiter', max_iter)
        options.set('tolfun', min_significant_change)
        #options.set('ftarget', target)

        # CMA-ES wants a single standard deviation as input, use the smallest
        # in the vector (if the user passed in a scalar, this will be the
        # value used).
        sigma0 = np.min(self._sigma0)

        # Tell cma-es to be quiet
        if not self._verbose:
            options.set('verbose', -9)

        # Start one or multiple runs
        best_solution = cma.optimization_tools.BestSolution()
        for i in xrange(1 + ipop):

            # Set population size, increase for ipop restarts
            options.set('popsize', n)
            if self._verbose:
                print('Run ' + str(1+i) + ': population size ' + str(n))
            n *= 2

            # Run repeats from random points
            #if i > 0:
            #    x0 = lower + brange * np.random.uniform(0, 1, d)

            # Search
            es = cma.CMAEvolutionStrategy(self._x0, sigma0, options)
            while not es.stop():
                candidates = es.ask()
                es.tell(candidates, evaluator.evaluate(candidates))
                if self._verbose:
                    es.disp()

            # Show result
            if self._verbose:
                es.result_pretty()

            # Update best solution
            best_solution.update(es.best)

        # Get best solution from all runs
        x, fx, evals = best_solution.get()

        # No result found? Then return hint and score of hint
        if x is None:
            return self._x0, self._function(hint)

        # Return proper result
        return x, fx

def cmaes(function, boundaries=None, x0=None, sigma0=None):
    """
    Runs a CMA-ES optimisation with the default settings.
    """
    return CMAES(function, boundaries, x0, sigma0).run()

