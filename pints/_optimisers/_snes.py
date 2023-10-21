#
# Seperable natural evolution strategy optimizer: SNES
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

        # Mean
        self._mu = np.array(self._x0, copy=True)

        # Best solution found
        self._x_best = pints.vector(x0)
        self._f_best = np.inf

        # Best guess of the solution is mu
        # We don't have f(mu), so we approximate it by max f(sample)
        self._f_guessed = np.inf

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

        # Boundaries? Then only pass user xs that are within bounds
        if self._boundaries is not None:
            self._user_ids = np.nonzero(
                [self._boundaries.check(x) for x in self._xs])
            self._user_xs = self._xs[self._user_ids]
            if len(self._user_xs) == 0:     # pragma: no cover
                warnings.warn(
                    'All points requested by SNES are outside the boundaries.')
        else:
            self._user_xs = self._xs

        # Set as read-only and return
        self._user_xs.setflags(write=False)
        return self._user_xs

    def f_best(self):
        """ See :meth:`Optimiser.f_best()`. """
        return self._f_best

    def f_guessed(self):
        """ See :meth:`Optimiser.f_guessed()`. """
        return self._f_guessed

    def _initialise(self):
        """
        Initialises the optimiser for the first iteration.
        """
        assert not self._running

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

        # Boundaries? Then reconstruct full fx vector
        if self._boundaries is not None and len(fx) < self._population_size:
            user_fx = fx
            fx = np.ones((self._population_size, )) * np.inf
            fx[self._user_ids] = user_fx

        # Order the normalized samples according to the scores
        order = np.argsort(fx)
        self._ss = self._ss[order]

        # Update center
        self._mu += self._eta_mu * self._sigmas * np.dot(self._us, self._ss)

        # Update variances
        self._sigmas *= np.exp(
            0.5 * self._eta_sigmas * np.dot(self._us, self._ss**2 - 1))

        # Update f_guessed on the assumption that the lowest value in our
        # sample approximates f(mu)
        self._f_guessed = fx[order[0]]

        # Update x_best and f_best
        if self._f_guessed < self._f_best:
            self._x_best = np.array(self._xs[order[0]], copy=True)
            self._f_best = fx[order[0]]

    def x_best(self):
        """ See :meth:`Optimiser.x_best()`. """
        return self._x_best

    def x_guessed(self):
        """ See :meth:`Optimiser.x_guessed()`. """
        return np.array(self._mu, copy=True)

