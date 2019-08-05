#
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#

from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import logging
import numpy as np
import pints


class SGD(pints.PopulationBasedOptimiser):
    """
    Finds the best parameters using the SGD method.
    SGD stands for Stochastic Gradient Descent.
    """

    def __init__(self, x0, sigma0=None, boundaries=None,
                 use_exact_grad=False):
        super(SGD, self).__init__(x0, sigma0, boundaries)

        # Set initial state
        self._running = False
        self._ready_for_tell = False

        # Best solution found
        self._xbest = pints.vector(x0)
        self._fbest = float('inf')

        # Python logger
        self._logger = logging.getLogger(__name__)
        self._population_size = 20

        # Do we use the approximate gradients?
        self._use_exact_grad = use_exact_grad

    def ask(self):
        """ See :meth:`Optimiser.ask()`. """
        # Initialise on first call
        if not self._running:
            self._initialise()

        # Ready for tell now
        self._ready_for_tell = True

        # New sample with all but one parameter fixed
        # Issue, our fbest and xbest are meaning less because we are not exploring on all
        # params at the same time.. (isn't a huge issue however)
        self._xs = np.array([]).reshape(0, self._n_parameters)
        zs = np.array([np.random.normal(1, 0.005)
                             for i in range(self._population_size)])
        for i in range(self._n_parameters):
            # Modify only for one parameter
            temp = np.full((self._population_size, self._n_parameters), self._weights)
            temp[:, i] *= zs
            self._xs = np.vstack([self._xs, temp])

        if self._manual_boundaries:
            # Manual boundaries? Then pass only xs that are within bounds
            self._user_ids = np.nonzero(
                [self._boundaries.check(x) for x in self._xs])
            self._user_xs = self._xs[self._user_ids]
            if len(self._user_xs) == 0:  # pragma: no cover
                self._logger.warning(
                    'All points requested by SGD are outside the boundaries.')
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
        assert (not self._running)

        self._step_size = 0.0000001

        self._manual_boundaries = False
        self._boundary_transform = None
        if isinstance(self._boundaries, pints.RectangularBoundaries):
            self._boundary_transform = pints.TriangleWaveTransform(
                self._boundaries)
        elif self._boundaries is not None:
            self._manual_boundaries = True

        self._weights = np.array(self._x0)

        # Update optimiser state
        self._running = True

    def name(self):
        """ See :meth:`Optimiser.name()`. """
        return 'Constant Step Stochastic Gradient Descent (SGD)'

    def running(self):
        """ See :meth:`Optimiser.running()`. """
        return self._running

    def set_step_size(self, step):
        self._step_size = step

    def gradient(self):
        return self._gradient

    def _suggested_population_size(self):
        """ See :meth:`Optimiser._suggested_population_size(). """
        return 4 + int(3 * np.log(self._n_parameters))

    def tell(self, fx):
        """ See :meth:`Optimiser.tell()`. """
        if not self._ready_for_tell:
            raise Exception('ask() not called before tell()')
        self._ready_for_tell = False

        if self._manual_boundaries and len(fx) < self._population_size:
            user_fx = fx
            fx = np.ones((self._population_size,)) * float('inf')
            fx[self._user_ids] = user_fx

        if self._use_exact_grad:
            grads = [f[1] for f in fx]
            fx = [f[0] for f in fx]
            gradient = self._exact_grad(grads)
        else:
            gradient = self._approx_grad(fx)
        order = np.argsort(fx)

        self._weights -= self._step_size * gradient * self._weights

        if fx[order[0]] < self._fbest:
            self._fbest = fx[order[0]]
            self._xbest = self._xs[order[0]]

    # This function approximates the gradient.
    # For each parameter we got through all possible combination of 2 samples and their scores to compute their
    # approximated gradient and add them together. Note that we are guaranteed to have at least self._repetition
    # 0 valued gradients
    def _approx_grad(self, scores):
        gradient = np.zeros(self._n_parameters)
        for i in range(self._n_parameters):
            for j in range(self._population_size):
                for k in range(self._population_size):
                    gradient[i] += (scores[j + i * self._population_size] - scores[k + i * self._population_size]) / \
                                   (self._user_xs[j + i * self._population_size][i] - self._user_xs[k + i * self._population_size][i])
        return gradient / (self._population_size * (self._population_size - 1))

    # This function takes the gradients on each timestep t as input
    # We are using the individual gradients and take the average to get an estimation of the real gradients for each
    # time step t
    def _exact_grad(self, grads):
        n = self._n_parameters
        p = self._population_size
        result = np.zeros(n)
        for j in range(p):
            result += grads[j]
        return result / p

    def xbest(self):
        """ See :meth:`Optimiser.xbest()`. """
        return self._xbest

    def weights(self):
        return self._weights
