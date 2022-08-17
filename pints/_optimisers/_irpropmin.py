#
# Improved Rprop local optimiser.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints

import numpy as np


class IRPropMin(pints.Optimiser):
    """
    iRprop- algorithm, as described in Figure 3 of [1]_.

    This is a local optimiser that requires gradient information, although it
    uses only the direction (sign) of the gradient in each dimension and
    ignores the magnitude. Instead, it maintains a separate step size for each
    dimension which grows when the sign of the gradient stays the same and
    shrinks when it changes.

    Pseudo-code is given below. Here ``p_j[i]`` denotes the j-th parameter at
    iteration ``i``, and ``df_j[i]`` is the corresponding derivative of the
    objective function (so both are scalars)::

        if df_j[i] * df_j[i - 1] > 0:
            step_size_j[i] = 1.2 * step_size_j[i-1]
        elif df_j[i] * df_j[i - 1] < 0:
            step_size_j[i] = 0.5 * step_size_j[i-1]
            df_j[i - 1] = 0
        p_j[i] = p_j[i] - sign(df_j[i]) * step_size_j[i]

    The line ``df_j[i - 1] = 0`` has two effects:

        1. It sets the update at this iteration to zero (using
           ``sign(df_j[i]) * step_size_j[i] = 0 * step_size_j[i]``).
        2. It ensures that the next iteration is performed (since
           ``df_j[i + 1] * df_j[i] = 0`` so neither if statement holds).

    In this implementation, the ``step_size`` is initialised as ``sigma_0``,
    the increase (0.5) & decrease factors (1.2) are fixed, and a minimum step
    size of ``1e-3 * min(sigma0)`` is enforced.

    This is an unbounded method: Any ``boundaries`` will be ignored.

    The name "iRprop-" was introduced by [1]_, and is a variation on the
    "Resilient backpropagation (Rprop)" optimiser introduced in [2]_.

    References
    ----------
    .. [1] Empirical Evaluation of the Improved Rprop Learning Algorithms.
           Igel and Hüsken, 2003, Neurocomputing
           https://doi.org/10.1016/S0925-2312(01)00700-7
    .. [2] A direct adaptive method for faster backpropagation learning: the
           RPROP algorithm. Riedmiller and Braun, 1993.
           https://doi.org/10.1109/ICNN.1993.298623

    """

    def __init__(self, x0, sigma0=0.1, boundaries=None):
        super().__init__(x0, sigma0, boundaries)

        # Set optimiser state
        self._running = False
        self._ready_for_tell = False

        # Best solution found
        self._x_best = self._x0
        self._f_best = float('inf')

        # Minimum and maximum step sizes
        self._step_min = 1e-3 * np.min(self._sigma0)

        # Current point, score, and gradient
        self._current = self._x0
        self._current_f = float('inf')
        self._current_df = None

        # Proposed next point (read-only, so can be passed to user)
        self._proposed = self._x0
        self._proposed.setflags(write=False)

        # Step size adaptations
        # TODO: Could be hyperparameters, but almost nobody varies these?
        self._eta_min = 0.5  # 0 < eta_min < 1
        self._eta_max = 1.2  # 1 < eta_max

        # Current step sizes
        self._step_size = np.array(self._sigma0)

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
        logger.add_float('Min. step')
        logger.add_float('Max. step')

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        logger.log(np.min(self._step_size))
        logger.log(np.max(self._step_size))

    def name(self):
        """ See :meth:`Optimiser.name()`. """
        return 'iRprop-'

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

        # First iteration
        if self._current_df is None:
            self._current_f = fx
            self._current_df = dfx
            self._f_best = fx
            self._x_best = self._current
            return

        # Get product of new and previous gradient
        dprod = dfx * self._current_df

        # Note: Could implement boundaries here by setting all dprod to < 0 if
        # the point is out of bounds?

        # Adapt step sizes
        self._step_size[dprod > 0] *= self._eta_max
        self._step_size[dprod < 0] *= self._eta_min

        # Bound step sizes
        if self._step_min is not None:
            self._step_size = np.maximum(self._step_size, self._step_min)
        # Note: Could implement step_max here if desired

        # Remove "weight backtracking"
        # This step ensures that, for each i where dprod < 0:
        #  1. p[i] will remain unchanged this iteration (sign(0) == 0)
        #  2. p[i] will change in the next iteration (dprod == 0), using
        #     the step size set in the current iteration
        dfx[dprod < 0] = 0

        # "Accept" proposed point
        self._current = self._proposed
        self._current_f = fx
        self._current_df = dfx

        # Take step in direction indicated by current gradient
        self._proposed = self._current - self._step_size * np.sign(dfx)
        self._proposed.setflags(write=False)

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

