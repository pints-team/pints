#
# Fixed learning-rate gradient descent.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import pints


class GradientDescent(pints.Optimiser):
    """
    Gradient-descent method with a fixed learning rate.
    """

    def __init__(self, x0, sigma0=0.1, boundaries=None):
        super(GradientDescent, self).__init__(x0, sigma0, boundaries)

        # Set optimiser state
        self._running = False
        self._ready_for_tell = False

        # Best solution found
        self._xbest = self._x0
        self._fbest = float('inf')

        # Learning rate
        self._eta = 0.01

        # Current point, score, and gradient
        self._current = self._x0
        self._current_f = None
        self._current_df = None

        # Proposed next point (read-only, so can be passed to user)
        self._proposed = self._x0
        self._proposed.setflags(write=False)

    def ask(self):
        """ See :meth:`Optimiser.ask()`. """

        # Running, and ready for tell now
        self._ready_for_tell = True
        self._running = True

        # Return proposed points (just the one)
        return [self._proposed]

    def fbest(self):
        """ See :meth:`Optimiser.fbest()`. """
        return self._fbest

    def learning_rate(self):
        """ Returns this optimiser's learning rate. """
        return self._eta

    def name(self):
        """ See :meth:`Optimiser.name()`. """
        return 'Gradient descent'

    def needs_sensitivities(self):
        """ See :meth:`Optimiser.needs_sensitivities()`. """
        return True

    def n_hyper_parameters(self):
        """ See :meth:`pints.TunableMethod.n_hyper_parameters()`. """
        return 1

    def running(self):
        """ See :meth:`Optimiser.running()`. """
        return self._running

    def set_hyper_parameters(self, x):
        """
        See :meth:`pints.TunableMethod.set_hyper_parameters()`.

        The hyper-parameter vector is ``[learning_rate]``.
        """
        self.set_learning_rate(x[0])

    def set_learning_rate(self, eta):
        """
        Sets the learning rate for this optimiser.

        Parameters
        ----------
        eta : float
            The learning rate, as a float greater than zero.
        """
        eta = float(eta)
        if eta <= 0:
            raise ValueError('Learning rate must greater than zero.')
        self._eta = eta

    def tell(self, reply):
        """ See :meth:`Optimiser.tell()`. """

        # Check ask-tell pattern
        if not self._ready_for_tell:
            raise Exception('ask() not called before tell()')
        self._ready_for_tell = False

        # Unpack reply
        fx, dfx = reply[0]

        # Move to proposed point
        self._current = self._proposed
        self._current_f = fx
        self._current_df = dfx

        # Propose next point
        self._proposed = self._current - self._eta * dfx
        self._proposed.setflags(write=False)

        # Update xbest and fbest
        if self._fbest > fx:
            self._fbest = fx
            self._xbest = self._current

    def xbest(self):
        """ See :meth:`Optimiser.xbest()`. """
        return self._xbest

