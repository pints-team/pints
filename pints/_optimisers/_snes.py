#
# Seperable natural evolution strategy optimizer: SNES
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


class SNES(pints.Optimiser):
    """
    *Extends:* :class:`Optimiser`

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
    def __init__(self, x0, sigma0=None, boundaries=None):
        super(SNES, self).__init__(x0, sigma0, boundaries)

        # Set initial state
        self._running = False
        self._ready_for_tell = False

        # Set default settings
        self.set_population_size()

        # Best solution found
        self._xbest = pints.vector(x0)
        self._fbest = float('inf')

    def ask(self):
        """See :meth:`Optimiser.ask()`."""
        # Initialise on first call
        if not self._running:
            self._initialise()

        # Ready for tell now
        self._ready_for_tell = True

        # Create new samples
        self._ss = np.array([np.random.normal(0, 1, self._dimension)
                            for i in range(self._population_size)])
        self._xs = self._mu + self._sigmas * self._ss

        # Perform boundary transform
        if self._boundaries:
            self._xs = self._xtransform(self._xs)

        # Set as read-only and return
        self._xs.setflags(write=False)
        return self._xs

    def fbest(self):
        """ See :meth:`Optimiser.fbest()`. """
        return self._fbest

    def _initialise(self):
        """
        Initialises the optimiser for the first iteration.
        """
        if self._running:
            raise Exception('Already initialised.')

        # Apply wrapper to implement boundaries
        if self._boundaries is not None:
            self._xtransform = pints.TriangleWaveTransform(self._boundaries)

        # Shorthands
        d = self._dimension
        n = self._population_size

        # Learning rates
        # TODO Allow changing before run() with method call
        self._eta_mu = 1
        # TODO Allow changing before run() with method call
        self._eta_sigmas = 0.2 * (3 + np.log(d)) * d ** -0.5

        # Pre-calculated utilities
        self._us = np.maximum(0, np.log(n / 2 + 1) - np.log(1 + np.arange(n)))
        self._us /= np.sum(self._us)
        self._us -= 1 / n

        # Center of distribution
        self._mu = np.array(self._x0, copy=True)

        # Initial square root of covariance matrix
        self._sigmas = np.array(self._sigma0, copy=True)

        # Update optimiser state
        self._running = True

    def mu(self):
        """
        Returns `mu`, the current estimate of the distribution's minimum.
        """
        if self._boundaries:
            return self._xtransform(self._mu)
        else:
            return np.array(self._mu, copy=True)

    def name(self):
        """See: :meth:`Optimiser.name()`."""
        return 'Seperable Natural Evolution Strategy (SNES)'

    def population_size(self):
        """ See :meth:`PopulationBasedOptimiser.population_size`. """
        return self._population_size

    def set_population_size(self, population_size=None, parallel=False):
        """ See :meth:`PopulationBasedOptimiser.set_population_size`. """
        if self._running:
            raise Exception('Cannot change settings during run.')

        # Check population size or set using heuristic
        if population_size is None:
            population_size = 4 + int(3 * np.log(self._dimension))
        else:
            population_size = int(population_size)
            if population_size < 1:
                raise ValueError('Population size must be at least 1.')

        # Round up to number of CPU cores
        if parallel:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            population_size = cpu_count * (
                ((population_size - 1) // cpu_count) + 1)

        # Store
        self._population_size = population_size

    def tell(self, fx):
        """See: :meth:`Optimiser.tell()`."""
        if not self._ready_for_tell:
            raise Exception('ask() not called before tell()')
        self._ready_for_tell = False

        # Order the normalized samples according to the scores
        order = np.argsort(fx)
        self._ss = self._ss[order]

        # Update center
        self._mu += self._eta_mu * self._sigmas * np.dot(self._us, self._ss)

        # Update variances
        self._sigmas *= np.exp(
            0.5 * self._eta_sigmas * np.dot(self._us, self._ss**2 - 1))

        # Update xbest and fbest
        # Note: The stored values are based on particles, not on the mean of
        # all particles! This has the advantage that we don't require an extra
        # evaluation at mu to get a pair (mu, f(mu)). The downside is that
        # xbest isn't the very best point. However, xbest and mu seem to
        # converge quite quickly, so that this difference disappears.
        if fx[order[0]] < self._fbest:
            self._xbest = self._xs[order[0]]
            self._fbest = fx[order[0]]

    def xbest(self):
        """ See :meth:`Optimiser.xbest`. """
        return self._xbest

