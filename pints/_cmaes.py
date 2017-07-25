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
        parallel = True
        
        # Number of IPOP repeats (0=no repeats, 1=repeat once=2 runs, etc.)
        ipop = 0
        
        # Parameter space dimension
        d = self._dimension
        
        # Population size
        # If parallel, round up to a multiple of the reported number of cores
        # In IPOP-CMAES, this will be used as the _initial_ population size
        n = 4 + int(3 * np.log(d))
        if parallel:
            cpu_count = multiprocessing.cpu_count()
            n = (((n - 1) // cpu_count) + 1) * cpu_count

        # Search is terminated after max_iter iterations
        max_iter = 10000
        # CMA-ES default: 100 + 50 * (d + 3)**2 // n**0.5 
        
        # Or if successive iterations do not produce a significant change
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
                [self._boundaries._lower, self._boundaries._upper])
        
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
        best_solution = cma.BestSolution()
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










'''
import cma
import numpy as np
import math
import multiprocessing
import itertools

def sample(v,data,model,prior,min_theta,max_theta,names):
    scaled_theta = abs(v[-1]) % 2
    if (scaled_theta <= 1):
        scaled_theta = min_theta + (max_theta-min_theta)*scaled_theta
    else:
        scaled_theta = max_theta - (max_theta-min_theta)*(scaled_theta-1)
    sample_params = prior.scale_sample_periodic(v[:-1],names)
    model.set_params_from_vector(sample_params,names)
    current,time = model.simulate(use_times=data.time)
    noise_log_likelihood = -math.log(scaled_theta)
    data_log_likelihood = data.log_likelihood(current,scaled_theta**2)/len(current)
    param_log_likelihood = prior.log_likelihood(sample_params,names)
    return -(noise_log_likelihood + data_log_likelihood + param_log_likelihood)


def sample_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return sample(*a_b)


def fit_model_with_cmaes(data, model, prior, IPOP=None):
    print '------------------------'
    print 'fitting model with cmaes'
    print '------------------------'
    x0 = np.zeros(prior.n+1)+0.5
    names = prior.get_parameter_names()

    max_current = np.max(data.current)
    min_theta = 0.005*max_current;
    max_theta = 0.03*max_current;


    if IPOP is None:
        es = cma.CMAEvolutionStrategy(x0, 0.25)
        print 'starting pool with',es.popsize,'workers'
        pool = multiprocessing.Pool(min(es.popsize,multiprocessing.cpu_count()-1))
        while not es.stop():
            X = es.ask()
            f_values = pool.map(sample_star,itertools.izip(X,
                                                 itertools.repeat(data),
                                                 itertools.repeat(model),
                                                 itertools.repeat(prior),
                                                 itertools.repeat(min_theta),
                                                 itertools.repeat(max_theta),
                                                 itertools.repeat(names)
                                                 ))
            # use chunksize parameter as es.popsize/len(pool)?
            es.tell(X, f_values)
            es.disp()
            #es.logger.add()
    else:
        bestever = cma.BestSolution()
        for lam in IPOP:
            print 'starting pool with',lam,'workers'
            pool = multiprocessing.Pool(min(lam,multiprocessing.cpu_count()-1))
            es = cma.CMAEvolutionStrategy(x0,  # 9-D
                                           0.25,  # initial std sigma0
                                           {'popsize': lam,  # options
                                            'verb_append': bestever.evalsall})
            while not es.stop():
                X = es.ask()    # get list of new solutions
                f_values = pool.map(sample_star,itertools.izip(X,
                                                 itertools.repeat(data),
                                                 itertools.repeat(model),
                                                 itertools.repeat(prior),
                                                 itertools.repeat(min_theta),
                                                 itertools.repeat(max_theta),
                                                 itertools.repeat(names)
                                                 ))

                es.tell(X, f_values) # besides for termination only the ranking in fit is used

                # display some output
                es.disp()  # uses option verb_disp with default 100

            print('termination:', es.stop())
            cma.pprint(es.best.__dict__)

            bestever.update(es.best)

    res = es.result()
    v = res[0]
    fitted_params = np.append(prior.scale_sample_periodic(v[:-1],names),[v[-1]*(max_theta-min_theta)+min_theta])
    objective_function = res[1]
    nevals = res[3]
    model.set_params_from_vector(fitted_params[:-1],names)
    print 'cmaes finished with objective function = ',objective_function,' and nevals = ',nevals, ' and noise estimate of ',fitted_params[-1]
    print '------------------------'
    return fitted_params
'''
