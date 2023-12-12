#
# Exponential natural evolution strategy optimizer: xNES
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
# Some code in this file was adapted from Myokit (see http://myokit.org)
#
import numpy as np
import pints
import scipy
import scipy.linalg
import warnings


class XNES(pints.PopulationBasedOptimiser):
    """
    Finds the best parameters using the xNES method described in [1]_, [2]_.

    xNES stands for Exponential Natural Evolution Strategy, and is
    designed for non-linear derivative-free optimization problems [1]_.

    Extends :class:`PopulationBasedOptimiser`.

    References
    ----------
    .. [1] Glasmachers, Schaul, Schmidhuber et al. (2010) "Exponential natural
           evolution strategies". Proceedings of the 12th annual conference on
           Genetic and evolutionary computation.
           https://doi.org/10.1145/1830483.1830557

    .. [2] PyBrain: The Python machine learning library
           http://pybrain.org
    """
    def __init__(self, x0, sigma0=None, boundaries=None):
        super(XNES, self).__init__(x0, sigma0, boundaries)

        # Set initial state
        self._running = False
        self._ready_for_tell = False

        # Samples
        self._zs = None       # Normalised samples
        self._xs = None       # De-normalised samples (mu + A dot zs)
        self._bounded_xs = None   # Subset of xs that are within the boundaries
        self._bounded_ids = None  # Indices of those xs

        # Normalisation / distribution
        self._mu = np.array(self._x0)   # Mean
        self._A = None                  # Covariance

        # Best solution seen
        self._x_best = pints.vector(x0)
        self._f_best = np.inf

        # Best guess of the solution is mu
        # We don't have f(mu), so we approximate it by min f(sample)
        self._f_guessed = np.inf

    def ask(self):
        """ See :meth:`Optimiser.ask()`. """
        # Initialise on first call
        if not self._running:
            self._initialise()

        # Ready for tell now
        self._ready_for_tell = True

        # Create new samples (normalised, and user values)
        self._zs = np.array([np.random.normal(0, 1, self._n_parameters)
                             for i in range(self._population_size)])
        self._xs = np.array([self._mu + np.dot(self._A, self._zs[i])
                             for i in range(self._population_size)])

        # Boundaries? Then only pass user xs that are within bounds
        if self._boundaries is not None:
            self._bounded_ids = np.nonzero(
                [self._boundaries.check(x) for x in self._xs])
            self._bounded_xs = self._xs[self._bounded_ids]
            if len(self._bounded_xs) == 0:     # pragma: no cover
                warnings.warn(
                    'All points requested by XNES are outside the boundaries.')
        else:
            self._bounded_xs = self._xs

        # Set as read-only and return
        self._bounded_xs.setflags(write=False)
        return self._bounded_xs

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
        self._eta_A = 0.6 * (3 + np.log(d)) * d ** -1.5

        # Pre-calculated utilities
        self._us = np.maximum(0, np.log(n / 2 + 1) - np.log(1 + np.arange(n)))
        self._us /= np.sum(self._us)
        self._us -= 1 / n

        # Center of distribution
        self._mu = np.array(self._x0, copy=True)

        # Initial square root of covariance matrix
        self._A = np.eye(d) * self._sigma0

        # Identity matrix of appropriate size
        self._I = np.eye(d)

        # Update optimiser state
        self._running = True

    def name(self):
        """ See :meth:`Optimiser.name()`. """
        return 'Exponential Natural Evolution Strategy (xNES)'

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
            bounded_fx = fx
            fx = np.ones((self._population_size, )) * np.inf
            fx[self._bounded_ids] = bounded_fx

        # Order the normalized samples according to the scores
        order = np.argsort(fx)
        self._zs = self._zs[order]

        # Update center
        Gd = np.dot(self._us, self._zs)
        self._mu += self._eta_mu * np.dot(self._A, Gd)

        # Update root of covariance matrix
        Gm = np.dot(
            np.array([np.outer(z, z).T - self._I for z in self._zs]).T,
            self._us)
        self._A *= scipy.linalg.expm(np.dot(0.5 * self._eta_A, Gm))

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

