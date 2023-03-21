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
    Adam optimiser (adaptive moment estimation), as described in [1]_ (see
    Algorithm 1).

    This method is a variation on gradient descent that maintains two
    "moments", allowing it to overshoot and go against the gradient for a short
    time. This property can make it more robust against noisy gradients.

    Pseudo-code is given below. Here the value of the j-th parameter at
    iteration i is given as ``p_j[i]`` and the corresponding derivative is
    denoted ``g_j[i]``::

        m_j[i] = beta1 * m_j[i - 1] + (1 - beta1) * g_j[i]
        v_j[i] = beta2 * v_j[i - 1] + (1 - beta2) * g_j[i]**2

        m_j' = m_j[i] / (1 - beta1**(1 + i))
        v_j' = v_j[i] / (1 - beta2**(1 + i))

        p_j[i] = p_j[i - 1] - alpha * m_j' / (sqrt(v_j') + eps)

    The initial values of the moments are ``m_j[0] = v_j[0] = 0``, after which
    they decay with rates ``beta1`` and ``beta2``. In this implementation,
    ``beta1 = 0.9`` and ``beta2 = 0.999``.

    The terms ``m_j'`` and ``v_j'`` are "initialisation bias corrected"
    versions of ``m_j`` and ``v_j`` (see section 3 of the paper).

    The parameter ``alpha`` is a step size, which is set as ``min(sigma0)`` in
    this implementation.

    Finally, ``eps`` is a small constant used to avoid division by zero, set to
    ``eps = 1e-8`` in this implementation.

    This is an unbounded method: Any ``boundaries`` will be ignored.

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

    def _log_init(self, logger):
        """ See :meth:`Loggable._log_init()`. """
        logger.add_float('b1')
        logger.add_float('b2')

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        logger.log(self._b1t)
        logger.log(self._b2t)

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

