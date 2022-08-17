#
# Adam optimiser.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints

import numpy as np


class Adam(pints.Optimiser):
    """
    Adam optimiser (adaptive moment estimation), as described in [1]_.

    This method is a variation on gradient descent that maintains two
    "moments", allowing it to overshoot and go against the gradient for a short
    time. This property can make it more robust against noisy gradients. Full
    pseudo-code is given in [1]_ (Algorithm 1).

    This implementation uses a fixed step size, set as `` min(sigma0)``. Note
    that the adaptivity in this method comes from the changing moments, not
    the step size.

    References
    ----------
    .. [1] Adam: A method for stochastic optimization
           Kingma and Ba, 2017, arxiv (version v9)
           https://doi.org/10.48550/arXiv.1412.6980
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

        # Moment vectors
        self._m = np.zeros(self._x0.shape)
        self._v = np.zeros(self._x0.shape)

        # Exponential decay rates for the moment estimates
        self._b1 = 0.9    # 0 < b1 <= 1
        self._b2 = 0.999  # 0 < b2 <= 1

        # Step size
        self._alpha = np.min(self._sigma0)

        # Small number added to avoid divide-by-zero
        self._eps = 1e-8

        # Powers of decay rates
        self._b1t = 1
        self._b2t = 1

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

        # Update bx^t
        self._b1t *= self._b1
        self._b2t *= self._b2

        # "Update biased first moment estimate"
        self._m = self._b1 * self._m + (1 - self._b1) * dfx

        # "Update biased secon raw moment estimate"
        self._v = self._b2 * self._v + (1 - self._b2) * dfx**2

        # "Compute bias-corrected first moment estimate"
        m = self._m / (1 - self._b1t)

        # "Compute bias-corrected second raw moment estimate"
        v = self._v / (1 - self._b2t)

        # Take step
        self._proposed = (
            self._current - self._alpha * m / (np.sqrt(v) + self._eps))

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

