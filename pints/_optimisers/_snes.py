#
# Seperable natural evolution strategy optimizer: SNES
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
# Some code in this file was adapted from Myokit (see http://myokit.org)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import logging
import numpy as np
import pints


class SNES(pints.PopulationBasedOptimiser):
    """
    Finds the best parameters using the SNES method described in [1]_, [2]_.

    SNES stands for Seperable Natural Evolution Strategy, and is designed for
    non-linear derivative-free optimization problems in high dimensions and
    with many local minima [1]_.

    It treats each dimension separately, making it suitable for higher
    dimensions.

    Extends :class:`PopulationBasedOptimiser`.

    References
    ----------
    .. [1] Schaul, Glasmachers, Schmidhuber (2011) "High dimensions and heavy
           tails for natural evolution strategies". Proceedings of the 13th
           annual conference on Genetic and evolutionary computation.
           https://doi.org/10.1145/2001576.2001692

    .. [2] PyBrain: The Python machine learning library
           http://pybrain.org
    """
    def __init__(self, x0, sigma0=None, boundaries=None):
        super(SNES, self).__init__(x0, sigma0, boundaries)

        # Set initial state
        self._running = False
        self._ready_for_tell = False

        # Best solution found
        self._xbest = pints.vector(x0)
        self._fbest = float('inf')

        # Python logger
        self._logger = logging.getLogger(__name__)

    def ask(self):
        """ See :meth:`Optimiser.ask()`. """
        # Initialise on first call
        if not self._running:
            self._initialise()

        # Ready for tell now
        self._ready_for_tell = True

        # Create new samples
        self._ss = np.array([np.random.normal(0, 1, self._n_parameters)
                            for i in range(self._population_size)])
        self._xs = self._mu + self._sigmas * self._ss

        # Create safe xs to pass to user
        if self._boundary_transform is not None:
            # Rectangular boundaries? Then perform boundary transform
            self._xs = self._boundary_transform(self._xs)
        if self._manual_boundaries:
            # Manual boundaries? Then pass only xs that are within bounds
            self._user_ids = np.nonzero(
                [self._boundaries.check(x) for x in self._xs])
            self._user_xs = self._xs[self._user_ids]
            if len(self._user_xs) == 0:     # pragma: no cover
                self._logger.warning(
                    'All points requested by SNES are outside the boundaries.')
        else:
            self._user_xs = self._xs

        # Set as read-only and return
        self._user_xs.setflags(write=False)
        return self._user_xs

    def fbest(self):
        """ See :meth:`Optimiser.fbest()`. """
        return self._fbest

    def _initialise(self):
        """
        Initialises the optimiser for the first iteration.
        """
        assert(not self._running)

        # Create boundary transform, or use manual boundary checking
        self._manual_boundaries = False
        self._boundary_transform = None
        if isinstance(self._boundaries, pints.RectangularBoundaries):
            self._boundary_transform = pints.TriangleWaveTransform(
                self._boundaries)
        elif self._boundaries is not None:
            self._manual_boundaries = True

        # Shorthands
        d = self._n_parameters
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

    def name(self):
        """ See :meth:`Optimiser.name()`. """
        return 'Seperable Natural Evolution Strategy (SNES)'

    def running(self):
        """ See :meth:`Optimiser.running()`. """
        return self._running

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
            user_fx = fx
            fx = np.ones((self._population_size, )) * float('inf')
            fx[self._user_ids] = user_fx

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
        """ See :meth:`Optimiser.xbest()`. """
        return self._xbest

