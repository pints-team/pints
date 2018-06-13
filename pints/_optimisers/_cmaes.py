#
# Uses the Python `cma` module to runs CMA-ES optimisations.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class CMAES(pints.PopulationBasedOptimiser):
    """
    *Extends:* :class:`Optimiser`

    Finds the best parameters using the CMA-ES method described in [1, 2] and
    implemented in the `cma` module.

    CMA-ES stands for Covariance Matrix Adaptation Evolution Strategy, and is
    designed for non-linear derivative-free optimization problems.

    [1] https://www.lri.fr/~hansen/cmaesintro.html

    [2] Hansen, Mueller, Koumoutsakos (2006) Reducing the time complexity of
    the derandomized evolution strategy with covariance matrix adaptation
    (CMA-ES).

    """
    def __init__(self, x0, sigma0=None, boundaries=None):
        super(CMAES, self).__init__(x0, sigma0, boundaries)

        # Set initial state
        self._running = False
        self._ready_for_tell = False

        # Best solution found
        self._xbest = pints.vector(x0)
        self._fbest = float('inf')

    def ask(self):
        """ See :meth:`Optimiser.ask()`. """
        # Initialise on first call
        if not self._running:
            self._initialise()

        # Ready for tell now
        self._ready_for_tell = True

        # Create new samples
        self._xs = np.array(self._es.ask())

        # Set as read-only and return
        self._xs.setflags(write=False)
        return self._xs

    def fbest(self):
        """ See :meth:`Optimiser.fbest()`. """
        f = self._es.result.fbest
        return float('inf') if f is None else f

    def _initialise(self):
        """
        Initialises the optimiser for the first iteration.
        """
        if self._running:
            raise Exception('Already initialised.')

        # Import cma (may fail!)
        # Only the first time this is called in a running program incurs
        # much overhead.
        import cma

        # Set up simulation
        options = cma.CMAOptions()

        # Set boundaries
        if self._boundaries is not None:
            options.set(
                'bounds',
                [list(self._boundaries._lower), list(self._boundaries._upper)]
            )

        # Set stopping criteria
        #options.set('maxiter', max_iter)
        #options.set('tolfun', min_significant_change)
        # options.set('ftarget', target)

        # Tell CMA not to worry about growing step sizes too much
        #options.set('tolfacupx', 10000)

        # CMA-ES wants a single standard deviation as input, use the smallest
        # in the vector (if the user passed in a scalar, this will be the
        # value used).
        self._sigma0 = np.min(self._sigma0)

        # Tell cma-es to be quiet
        options.set('verbose', -9)

        # Set population size
        options.set('popsize', self._population_size)

        # CMAES always seeds np.random, whether you ask it too or not, so to
        # get consistent debugging output, we should always pass in a seed.
        # Instead of using a fixed number (which would be bad), we can use a
        # randomly generated number: This will ensure pseudo-randomness, but
        # produce consistent results if np.random has been seeded before
        # calling.
        options.set('seed', np.random.randint(2**32))

        # Search
        self._es = cma.CMAEvolutionStrategy(self._x0, self._sigma0, options)

        # Update optimiser state
        self._running = True

    def name(self):
        """ See :meth:`Optimiser.name()`. """
        return 'Covariance Matrix Adaptation Evolution Strategy (CMA-ES)'

    def running(self):
        """ See :meth:`Optimiser.running()`. """
        return self._running

    def stop(self):
        """ See :meth:`Optimiser.stop()`. """
        if not self._running:
            return False

        # CMAES Stopping conditions:
        # tolconditioncov(1e14) ERROR CHECK: Condition of covariance matrix
        # tolfacupx(1e3) ERROR CHECK: Massive increases in step-size
        # timeout(inf) Limits real time
        # tolupsigma(1e20) Creeping / slow improvement
        # tolstagnation(has default) Threshold in unchanged iterations
        # tolx(1e-11) Threshold on change in position
        # ftarget(no default) Threshold on target function value
        # tolfun(1e-11) Threshold on change in target value (one iteration)
        # tolfunhist(1e-12) Threshold on long-term change in target value
        # maxfevals(inf) Max function evaluations
        # maxiter(has default) Max iterations
        # noeffectcoord ?
        # noeffectaxis ?
        # flat fitness: CMAES thinks the landscape is flat
        # "||xmean||^2<ftarget"
        # callback: User callback triggered stop
        stop = self._es.stop()
        if 'tolconditioncov' in stop:
            return 'Ill-conditioned covariance matrix.'
        return False

    def _suggested_population_size(self):
        """ See :meth:`Optimiser._suggested_population_size(). """
        return 4 + int(3 * np.log(self._dimension))

    def tell(self, fx):
        """ See :meth:`Optimiser.tell()`. """
        if not self._ready_for_tell:
            raise Exception('ask() not called before tell()')
        self._ready_for_tell = False

        # Tell CMA-ES
        self._es.tell(self._xs, fx)

    def xbest(self):
        """ See :meth:`Optimiser.xbest()`. """
        x = self._es.result.xbest
        return self._x0 if x is None else x

