#
# AdaDelta optimiser.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints

import numpy as np


class AdaDelta(pints.Optimiser):
    """
    AdaDelta variant of AdaGrad by [1]_, as given in [1,3]_.

    Pseudo code is given below. Here, ``p_j[i]`` denotes the j-th parameter at
    iteration i, while ``g_j[i]`` is the gradient with respect to parameter j.


        v_j[i] = rho * v_j[i - 1] + (1 - rho) * g_j[i]**2
        d = sqrt((w_j[i - 1] + eps) / (v_j[i] + eps)) * g_j[i]
        w_j[i] = rho * w_j[i - 1] + (1 - rho) * d**2
        p_j[i] = p_j[i - 1] - d

    Here ``v_j[0] = 0`` and ``w_j[0] = 0, ``rho`` is a constant decay rate,
    and ``eps`` is a small number used to avoid numerical errors.

    In this implementation, ``eps = 1e-6`` and ``rho = 0.95``. Note that there
    is no learning rate hyperparameter in this algorithm.

    Note: Boundaries and the value of ``sigma0`` are ignored.

    References
    ----------
    .. [1] ADADELTA: An adaptive Learning Rate Method.
           Zeiler, 2012. arXiv.
           https://arxiv.org/abs/1212.5701

    .. [3] An overview of gradient descent optimization algorithms.
           Ruder, 2016. arXiv
           https://arxiv.org/abs/1609.04747

    """

    def __init__(self, x0, sigma0=0.1, boundaries=None):
        super().__init__(x0, sigma0, boundaries)

        # Set optimiser state
        self._running = False
        self._ready_for_tell = False

        # Best solution found
        self._x_best = self._x0
        self._f_best = np.inf

        # Current point, score, and gradient
        self._current = self._x0
        self._current_f = np.inf
        self._current_df = None

        # Proposed next point (read-only, so can be passed to user)
        self._proposed = self._x0
        self._proposed.setflags(write=False)

        # Online sum of gradient**2 and d**2
        self._v = np.zeros(self._x0.shape)
        self._w = np.zeros(self._x0.shape)

        # Decay parameter
        self._rho = 0.95

        # Small number added to avoid divide-by-zero
        self._eps = 1e-6

        # Step size
        self._alpha = np.min(self._sigma0)

    def ask(self):
        """ See :meth:`Optimiser.ask()`. """

        # Running, and ready for tell now
        self._ready_for_tell = True
        self._running = True

        # Return proposed points (just the one)
        return [self._proposed]

    def f_best(self):
        """ See :meth:`Optimiser.f_best()`. """
        return self._f_best

    def f_guessed(self):
        """ See :meth:`Optimiser.f_guessed()`. """
        return self._current_f

    def name(self):
        """ See :meth:`Optimiser.name()`. """
        return 'Adam'

    def needs_sensitivities(self):
        """ See :meth:`Optimiser.needs_sensitivities()`. """
        return True

    def n_hyper_parameters(self):
        """ See :meth:`pints.TunableMethod.n_hyper_parameters()`. """
        return 0

    def running(self):
        """ See :meth:`Optimiser.running()`. """
        return self._running

    def tell(self, reply):
        """ See :meth:`Optimiser.tell()`. """

        # Check ask-tell pattern
        if not self._ready_for_tell:
            raise Exception('ask() not called before tell()')
        self._ready_for_tell = False

        # Unpack reply
        fx, dfx = reply[0]

        # Update current point
        self._current = self._proposed
        self._current_f = fx
        self._current_df = dfx

        # Accumulate gradients
        self._v = self._rho * self._v + (1 - self._rho) * dfx**2

        # Calculate update
        d  = np.sqrt((self._w + self._eps) / (self._v + self._eps)) * dfx

        # Accumulate updates
        self._w = self._rho * self._w + (1 - self._rho) * d**2

        # Take step
        self._proposed = self._current - d

        # Update x_best and f_best
        if self._f_best > fx:
            self._f_best = fx
            self._x_best = self._current

    def x_best(self):
        """ See :meth:`Optimiser.x_best()`. """
        return self._x_best

    def x_guessed(self):
        """ See :meth:`Optimiser.x_guessed()`. """
        return self._current

