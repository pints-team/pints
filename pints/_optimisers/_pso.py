#
# Particle swarm optimisation (PSO).
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
# Some code in this file was adapted from Myokit (see http://myokit.org)
#
import numpy as np
import pints
import warnings


class PSO(pints.PopulationBasedOptimiser):
    """
    Finds the best parameters using the PSO method described in [1]_.

    Particle Swarm Optimisation (PSO) is a global search method (so refinement
    with a local optimiser is advised!) that works well for problems in high
    dimensions and with many local minima. Because it treats each parameter
    independently, it does not require preconditioning of the search space.

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

    Extends :class:`PopulationBasedOptimiser`.

    References
    ----------
    .. [1] Kennedy, Eberhart (1995) Particle Swarm Optimization.
           IEEE International Conference on Neural Networks
           https://doi.org/10.1109/ICNN.1995.488968
    """

    def __init__(self, x0, sigma0=None, boundaries=None):
        super(PSO, self).__init__(x0, sigma0, boundaries)

        # Set initial state
        self._running = False
        self._ready_for_tell = False

        # Set default settings
        self.set_local_global_balance()

    def ask(self):
        """ See :meth:`Optimiser.ask()`. """
        # Initialise on first call
        if not self._running:
            self._initialise()

        # Ready for tell now
        self._ready_for_tell = True

        # Return a copy of the filtered points (copy is used so that 1. the
        # user cannot modify the points and mess up the state, and 2. so that
        # the user can store the points without us modifying them).
        return np.copy(self._user_xs)

    def f_best(self):
        """ See :meth:`Optimiser.f_best()`. """
        return self._fg if self._running else np.inf

    def _initialise(self):
        """
        Initialises the optimiser for the first iteration.
        """
        assert not self._running

        # Initialize swarm
        self._xs = []     # Particle coordinate vectors
        self._vs = []     # Particle velocity vectors
        self._fl = []     # Best local score
        self._pl = []     # Best local position

        # Set initial positions
        self._xs.append(np.array(self._x0, copy=True))

        # Attempt to sample n - 1 points from the boundaries
        if self._boundaries is not None:
            try:
                self._xs.extend(
                    self._boundaries.sample(self._population_size - 1))
            except NotImplementedError:
                # Not all boundaries implement sampling.
                pass

        # If we couldn't sample from the boundaries, use gaussian sampling
        # around x0.
        if len(self._xs) < self._population_size:
            for i in range(self._population_size - 1):
                self._xs.append(np.random.normal(self._x0, self._sigma0))
        self._xs = np.array(self._xs)

        # Set initial velocities
        for i in range(self._population_size):
            self._vs.append(1e-1 * self._sigma0 *
                            np.random.uniform(0, 1, self._n_parameters))

        # Set initial scores and local best
        for i in range(self._population_size):
            self._fl.append(np.inf)
            self._pl.append(self._xs[i])

        # Set global best position and score
        self._fg = np.inf
        self._pg = self._xs[0]

        # Boundaries? Then filter out out-of-bounds points from xs
        self._user_xs = self._xs
        if self._boundaries is not None:
            self._user_ids = np.nonzero(
                [self._boundaries.check(x) for x in self._xs])
            self._user_xs = self._xs[self._user_ids]
            if len(self._user_xs) == 0:     # pragma: no cover
                warnings.warn(
                    'All initial PSO particles are outside the boundaries.')

        # Set local/global exploration balance
        self.set_local_global_balance()

        # Update optimiser state
        self._running = True

    def _log_init(self, logger):
        """ See :meth:`Loggable._log_init()`. """
        # Show best position of each particle
        for i in range(self._population_size):
            logger.add_float('f' + str(i), file_only=True)

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        # Show best position of each particle
        for f in self._fl:
            logger.log(f)

    def name(self):
        """ See :meth:`Optimiser.name()`. """
        return 'Particle Swarm Optimisation (PSO)'

    def running(self):
        """ See :meth:`Optimiser.running()`. """
        return self._running

    def set_local_global_balance(self, r=0.5):
        """
        Set the balance between local and global exploration for each particle,
        using a parameter ``r`` such that ``r = 1`` is a fully local search and
        ``r = 0`` is a fully global search.
        """
        if self._running:
            raise Exception('Cannot change settings during run.')

        # Check r
        r = float(r)
        if r < 0 or r > 1:
            raise ValueError('Parameter r must be in the range 0-1.')

        # Set almax and agmax based on r
        _amax = 4.1
        self._almax = r * _amax
        self._agmax = _amax - self._almax

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 2

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[population_size,
        local_global_balance]``.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_population_size(x[0])
        self.set_local_global_balance(x[1])

    def _suggested_population_size(self):
        """ See :meth:`Optimiser._suggested_population_size(). """
        return 4 + int(3 * np.log(self._n_parameters))

    def tell(self, fx):
        """ See :meth:`Optimiser.tell()`. """
        if not self._ready_for_tell:
            raise Exception('ask() not called before tell()')
        self._ready_for_tell = False

        # Boundaries? Then reconstruct full fx vector
        if self._boundaries is not None and len(fx) < self._population_size:
            user_fx = fx
            fx = np.ones((self._population_size, )) * np.inf
            fx[self._user_ids] = user_fx

        # Update particles
        for i in range(self._population_size):

            # Update best local position and score
            if fx[i] < self._fl[i]:
                self._fl[i] = fx[i]
                self._pl[i] = np.array(self._xs[i], copy=True)

            # Calculate "velocity"
            al = np.random.uniform(0, self._almax, self._n_parameters)
            ag = np.random.uniform(0, self._agmax, self._n_parameters)
            self._vs[i] += (
                al * (self._pl[i] - self._xs[i]) +
                ag * (self._pg - self._xs[i]))

            # Reduce speed if going too fast, as indicated by going out of
            # bounds.
            # This is not in the original algorithm but seems to work well
            if self._boundaries is not None:
                if not self._boundaries.check(self._xs[i] + self._vs[i]):
                    self._vs[i] *= 0.5

            # Update position
            self._xs[i] += self._vs[i]

        # Boundaries? Then filter out out-of-bounds points from xs
        self._user_xs = self._xs
        if self._boundaries is not None:
            self._user_ids = np.nonzero(
                [self._boundaries.check(x) for x in self._xs])
            self._user_xs = self._xs[self._user_ids]
            if len(self._user_xs) == 0:     # pragma: no cover
                warnings.warn('All PSO particles are outside the boundaries.')

        # Update global best score
        i = np.argmin(self._fl)
        if self._fl[i] < self._fg:
            self._fg = self._fl[i]
            self._pg = np.array(self._pl[i], copy=True)

    def x_best(self):
        """ See :meth:`Optimiser.x_best()`. """
        if self._running:
            return np.array(self._pg, copy=True)
        return np.array(self._x0, copy=True)

