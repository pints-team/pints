#
# Particle swarm optimisation (PSO).
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

class PSO(pints.Optimiser):
    """
    Finds the best parameters using the PSO method described in [1].
    
    Particle Swarm Optimisation (PSO) is a global search method (so refinement
    with a local optimiser is advised!) that works well for problems in high
    dimensions and with many local minima. Because it treats each parameter
    indpendently, it does not require preconditioning of the search space.
    
    Detailed description:
    
    In a particle swarm optimization, the parameter space is explored by ``n``
    independent particles. The particles perform a pseudo-random walk through
    the parameter space, guided by their own personal best score and the global
    optimum found so far.
        
    The method starts by creating a swarm of ``n`` particles and assigning each
    an initial position and initial velocity (see the explanation of the
    arguments ``hints`` and ``v`` for details). Each particle's score is
    calculated and set as the particle's current best local score ``pl``. The
    best score of all the particles is set as the best global score ``pg``.
    
    Next, an iterative procedure is run that updates each particle's velocity
    ``v`` and position ``x`` using::
    
        v[k] = v[k-1] + al * (pl - x[k-1]) + ag * (pg - x[k-1])
        x[k] = v[k]
        
    Here, ``x[t]`` is the particle's current position and ``v[t]`` its current
    velocity. The values ``al`` and ``ag`` are scalars randomly sampled from a
    uniform distribution, with values bound by ``r * 4.1`` and
    ``(1 - r) * 4.1``. Thus a swarm with ``r = 1`` will only use local
    information, while a swarm with ``r = 0`` will only use global information.
    The de facto standard is ``r = 0.5``. The random sampling is done each time
    ``al`` and ``ag`` are used: at each time step every particle performs ``m``
    samplings, where ``m`` is the dimensionality of the search space.
    
    Pseudo-code algorithm::
    
        almax = r * 4.1
        agmax = 4.1 - almax
        while stopping criterion not met:
            for i in [1, 2, .., n]:
                if f(x[i]) < f(p[i]):
                    p[i] = x[i]
                pg = min(p[1], p[2], .., p[n])
                for j in [1, 2, .., m]:
                    al = uniform(0, almax)
                    ag = uniform(0, agmax)
                    v[i,j] += al * (p[i,j] - x[i,j]) + ag * (pg[i,j]  - x[i,j])
                    x[i,j] += v[i,j]
    
    References:
    
    [1] Kennedy, Eberhart (1995) Particle Swarm Optimization.
    IEEE International Conference on Neural Networks
   
    """
    def run(self):

        # Default search parameters
        parallel = True
        
        # Global/local search balance
        r = 0.5
        
        # Search is terminated after max_iter iterations
        max_iter = 10000

        # Or if the result doesn't change significantly for a while
        max_unchanged_iterations = 200
        min_significant_change = 1e-11
        unchanged_iterations = 0

        # Parameter space dimension
        d = self._dimension

        # Population size
        #TODO: Allow this to be set by the user!
        # If parallel, round up to a multiple of the reported number of cores
        n = 4 + int(3 * np.log(d))
        if parallel:            
            cpu_count = multiprocessing.cpu_count()
            n = min(3, (((n - 1) // cpu_count) + 1)) * cpu_count

        # Set up progress reporting in verbose mode
        nextMessage = 0
        if self._verbose:
            if parallel:
                print('Running in parallel mode with population size '
                    + str(n))
            else:
                print('Running in sequential mode with population size '
                    + str(n))

        # Set parameters based on global/local balance r
        amax = 4.1
        almax = r * amax
        agmax = amax - almax
        
        # Initialize swarm
        xs = [] # Particle coordinate vectors
        vs = [] # Particle velocity vectors
        fs = [] # Particle scores
        fl = [] # Best local score
        pl = [] # Best local position
        fg = 0  # Best global score
        pg = 0  # Best global position
        
        # Set initial positions
        xs.append(np.array(self._x0, copy=True))
        if self._boundaries is None:
            for i in xrange(1, n):
                xs.append(np.random.normal(self._x0, self._sigma0))
        else:
            for i in xrange(1, n):
                xs.append(self._boundaries._lower + np.random.uniform(0, 1, d)
                    * (self._boundaries._upper - self._boundaries._lower))

        # Set initial velocities
        for i in xrange(n):
            vs.append(self._sigma0 * np.random.uniform(0, 1, d))    
        
        # Set initial scores and local best
        for i in xrange(n):
            fs.append(float('inf'))
            fl.append(float('inf'))
            pl.append(xs[i])

        # Set global best position and score
        fg = float('inf')
        pg = xs[0]

        # Apply wrapper to score function to implement boundaries
        function = self._function
        if self._boundaries is not None:
            function = pints.InfBoundaryTransform(function, self._boundaries)

        # Create evaluator object
        if parallel:
            evaluator = pints.ParallelEvaluator(self._function)
        else:
            evaluator = pints.SequentialEvaluator(self._function)
        
        # Start searching
        for iteration in xrange(1, 1 + max_iter):
            
            # Calculate scores
            fs = evaluator.evaluate(xs)
            
            # Update particles
            for i in xrange(n):
                # Update best local position and score
                if fs[i] < fl[i]:
                    fl[i] = fs[i]
                    pl[i] = np.array(xs[i], copy=True)

                # Calculate "velocity"
                al = np.random.uniform(0, almax, d)
                ag = np.random.uniform(0, agmax, d)            
                vs[i] += al * (pl[i] - xs[i]) + ag * (pg - xs[i])

                # Update position
                e = xs[i] + vs[i]
                if self._boundaries is not None:
                    # To reduce the amount of time spent outside the bounds of
                    # the search space, the velocity of any particle outside
                    # the bounds is reduced by a factor
                    #  (1 / (1 + number of boundary violations)).
                    if not self._boundaries.check(e):
                        vs[i] *= 1 / (1 + np.sum(e))
                xs[i] += vs[i]

            # Update global best
            i = np.argmin(fl)
            if fl[i] < fg:
                # Check if this counts as a significant change
                fnew = fl[i]
                if np.sum(np.abs(fnew - fg)) < min_significant_change:
                    unchanged_iterations += 1
                else:
                    unchanged_iterations = 0
            
                # Update best
                fg = fnew
                pg = np.array(pl[i], copy=True)
            else:
                unchanged_iterations += 1

            # Show progress in verbose mode:
            if self._verbose and iteration >= nextMessage:
                print(str(iteration) + ': ' + str(fg))
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

        # Show final value
        if self._verbose:
            print(str(iteration) + ': ' + str(fg))

        # Return best position and score
        return pg, fg

def pso(function, boundaries=None, x0=None, sigma0=None):
    """
    Runs a PSO optimisation with the default settings.
    """
    return PSO(function, boundaries, x0, sigma0).run() 




