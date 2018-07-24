#
# Particle swarm optimisation (PSO).
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
# Some code in this file was adapted from Myokit (see http://myokit.org)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class PSO(pints.PopulationBasedOptimiser):
    """
    *Extends:* :class:`PopulationBasedOptimiser`

    Finds the best parameters using the PSO method described in [1].

    Particle Swarm Optimisation (PSO) is a global search method (so refinement
    with a local optimiser is advised!) that works well for problems in high
    dimensions and with many local minima. Because it treats each parameter
    independently, it does not require preconditioning of the search space.

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

        # Return points
        return self._user_xs

    def fbest(self):
        """ See :meth:`Optimiser.fbest()`. """
        if self._running:
            return self._fg
        return float('inf')

    def _initialise(self):
        """
        Initialises the optimiser for the first iteration.
        """
        assert(not self._running)

        # Initialize swarm
        self._xs = []     # Particle coordinate vectors
        self._vs = []     # Particle velocity vectors
        self._fl = []     # Best local score
        self._pl = []     # Best local position

        # Set initial positions
        self._xs.append(np.array(self._x0, copy=True))
        if self._boundaries is None:
            for i in range(1, self._population_size):
                self._xs.append(np.random.normal(self._x0, self._sigma0))
        else:
            for i in range(1, self._population_size):
                self._xs.append(
                    self._boundaries._lower
                    + np.random.uniform(0, 1, self._dimension)
                    * (self._boundaries._upper - self._boundaries._lower))
        self._xs = np.array(self._xs, copy=True)

        # Set initial velocities
        for i in range(self._population_size):
            self._vs.append(
                1e-1 * self._sigma0 * np.random.uniform(0, 1, self._dimension))

        # Set initial scores and local best
        for i in range(self._population_size):
            self._fl.append(float('inf'))
            self._pl.append(self._xs[i])

        # Set global best position and score
        self._fg = float('inf')
        self._pg = self._xs[0]

        # Create boundary transform
        self._boundary_transform = None
        if self._boundaries is not None:
            self._boundary_transform = pints.TriangleWaveTransform(
                self._boundaries)

        # Create safe xs to pass to user
        self._user_xs = np.array(self._xs, copy=True)

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
        using a parameter `r` such that `r = 1` is a fully local search and
        `r = 0` is a fully global search.
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
        return 4 + int(3 * np.log(self._dimension))

    def tell(self, fx):
        """ See :meth:`Optimiser.tell()`. """
        if not self._ready_for_tell:
            raise Exception('ask() not called before tell()')
        self._ready_for_tell = False

        # Update particles
        for i in range(self._population_size):

            # Update best local position and score
            if fx[i] < self._fl[i]:
                self._fl[i] = fx[i]
                self._pl[i] = np.array(self._xs[i], copy=True)

            # Calculate "velocity"
            al = np.random.uniform(0, self._almax, self._dimension)
            ag = np.random.uniform(0, self._agmax, self._dimension)
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

        # Map points outside of the search space back into it
        if self._boundary_transform is not None:
            self._xs = self._boundary_transform(self._xs)

        # Update global best
        i = np.argmin(self._fl)
        if self._fl[i] < self._fg:
            self._fg = self._fl[i]
            self._pg = np.array(self._pl[i], copy=True)

        # Create safe xs to pass to user
        self._user_xs = np.array(self._xs, copy=True)

    def xbest(self):
        """ See :meth:`Optimiser.xbest()`. """
        if self._running:
            return np.array(self._pg, copy=True)
        return np.array(self._x0, copy=True)
