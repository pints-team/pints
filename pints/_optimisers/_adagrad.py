#
# AdaGrad optimiser.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints

import numpy as np


class AdaGrad(pints.Optimiser):
    """
    AdaGrad optimiser by [1]_, as given in [2]_.




    References
    ----------
    .. [1] Adaptive subgradient methods for online learning and stochastic
           optimization. Duchi, Hazan, and Singer, 2011.
           Journal of Machine Learning Research
           https://dl.acm.org/doi/10.5555/1953048.2021068

    .. [2] Defossez, Bottou, Bach, Usunier (2020) A Simple Convergence Proof of Adam and Adagrad

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

        # Online sum of gradient**2
        self._v = np.zeros(self._x0.shape)

        # Small number added to avoid divide-by-zero
        self._eps = 1e-8

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

        # Update online sum of gradient**2
        self._v += dfx**2

        # Take step
        self._proposed = (
            self._current - self._alpha * dfx / np.sqrt(self._v + self._eps))

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

