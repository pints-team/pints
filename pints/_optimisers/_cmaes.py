#
# Uses the Python ``cma`` module to run CMA-ES optimisations.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
import pints
import warnings


class CMAES(pints.PopulationBasedOptimiser):
    """
    Finds the best parameters using the CMA-ES method described in [1]_, [2]_
    and implemented in the ``cma`` module [3]_.

    CMA-ES stands for Covariance Matrix Adaptation Evolution Strategy, and is
    designed for non-linear derivative-free optimization problems.

    Extends :class:`PopulationBasedOptimiser`.

    References
    ----------
    .. [1] The CMA Evolution Strategy: A Tutorial
           Nikolaus Hanse, arxiv
           https://arxiv.org/abs/1604.00772

    .. [2] Hansen, Mueller, Koumoutsakos (2006) "Reducing the time complexity
           of the derandomized evolution strategy with covariance matrix
           adaptation (CMA-ES)". Evolutionary Computation
           https://doi.org/10.1162/106365603321828970

    .. [3] PyPi page for ``cma``
           https://pypi.org/project/cma/
    """

    def __init__(self, x0, sigma0=None, boundaries=None):
        super(CMAES, self).__init__(x0, sigma0, boundaries)

        # Set initial state
        self._running = False
        self._ready_for_tell = False

        # Estimated f(x_guessed)
        self._f_guessed = np.inf

    def ask(self):
        """ See :meth:`Optimiser.ask()`. """
        # Initialise on first call
        if not self._running:
            self._initialise()

        # Ready for tell now
        self._ready_for_tell = True

        # Create new samples
        self._user_xs = self._xs = np.array(self._es.ask())

        # Manual boundaries? Then filter out points that are out of bounds
        if self._manual_boundaries:
            self._user_ids = np.nonzero(
                [self._boundaries.check(x) for x in self._xs])
            self._user_xs = self._xs[self._user_ids]
            if len(self._user_xs) == 0:     # pragma: no cover
                warnings.warn(
                    'All points requested by CMA-ES are outside the'
                    ' boundaries.')

        # Set as read-only and return
        self._user_xs.setflags(write=False)
        return self._user_xs

    def f_best(self):
        """ See :meth:`Optimiser.f_best()`. """
        f = self._es.result.fbest if self._running else None
        return np.inf if f is None else f

    def f_guessed(self):
        """ See :meth:`Optimiser.f_guessed()`. """
        return self._f_guessed

    def _initialise(self):
        """
        Initialises the optimiser for the first iteration.
        """
        assert not self._running

        # Import cma (may fail!)
        # Only the first time this is called in a running program incurs
        # much overhead.
        import cma

        # Set up simulation
        options = cma.CMAOptions()

        # Set boundaries, or use manual boundary checking
        self._manual_boundaries = False
        if isinstance(self._boundaries, pints.RectangularBoundaries):
            options.set(
                'bounds',
                [list(self._boundaries._lower), list(self._boundaries._upper)]
            )
        elif self._boundaries is not None:
            self._manual_boundaries = True

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
        options.set('seed', np.random.randint(2**31))

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
        if stop:
            if 'tolconditioncov' in stop:    # pragma: no cover
                return 'Ill-conditioned covariance matrix.'

            # self._logger.debug(
            #    'CMA-ES stopping condition(s) reached: ' +
            #    '; '.join([str(x) for x in stop.keys()]))

        return False

    def _suggested_population_size(self):
        """ See :meth:`Optimiser._suggested_population_size(). """
        return 4 + int(3 * np.log(self._n_parameters))

    def tell(self, fx):
        """ See :meth:`Optimiser.tell()`. """
        if not self._ready_for_tell:
            raise Exception('ask() not called before tell()')
        self._ready_for_tell = False

        # Manual boundaries? Then reconstruct full fx vector
        if self._manual_boundaries and len(fx) < self._population_size:
            # Note: CMA-ES uses ``nan`` to mean "could not calculate this
            # point". But for some reason this doesn't work well, causes a lot
            # of points to go out of bounds for some reason. Works much better
            # when just using ``inf``...
            user_fx = fx
            fx = np.ones((self._population_size, )) * np.inf
            fx[self._user_ids] = user_fx

        # Tell CMA-ES
        self._es.tell(self._xs, fx)

        # Update f_guessed, on the assumption that the best value in our
        # current set of points is a reasonable approximation of f(mu). This
        # will become more true as the optimiser progresses.
        self._f_guessed = np.min(fx)

    def x_best(self):
        """ See :meth:`Optimiser.x_best()`. """
        x = self._es.result.xbest if self._running else None
        return np.array(self._x0 if x is None else x)

    def x_guessed(self):
        """ See :meth:`Optimiser.x_guessed()`. """
        x = self._es.result.xfavorite if self._running else None
        return np.array(self._x0 if x is None else x)

