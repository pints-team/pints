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

    The name "iRprop-" was introduced by [1]_, and is a variation on the
    "Resilient backpropagation (Rprop)" optimiser introduced in [2]_.

    This is a local optimiser that requires gradient information, although it
    uses only the direction (sign) of the gradient in each dimension and
    ignores the magnitude. Instead, it maintains a separate step size for each
    dimension which grows when the sign of the gradient stays the same and
    shrinks when it changes.

    Pseudo-code is given below. Here ``p_j[i]`` denotes the j-th parameter at
    iteration ``i``, and ``df_j[i]`` is the corresponding derivative of the
    objective function (so both are scalars)::

        if df_j[i] * df_j[i - 1] > 0:
            step_size_j[i] = 1.2 * step_size_j[i - 1]
        elif df_j[i] * df_j[i - 1] < 0:
            step_size_j[i] = 0.5 * step_size_j[i - 1]
            df_j[i] = 0
        step_size_j[i] = min(max(step_size_j[i], min_step_size), max_step_size)
        p_j[i] = p_j[i] - sign(df_j[i]) * step_size_j[i]

    The line ``df_j[i] = 0`` has two effects:

        1. It sets the update at this iteration to zero (using
           ``sign(df_j[i]) * step_size_j[i] = 0 * step_size_j[i]``).
        2. It ensures that the next iteration is performed (since
           ``df_j[i] * df_j[i - 1] == 0`` so neither if-statement holds).

    In this implementation, the initial ``step_size`` is set to ``sigma0``, the
    default minimum step size is set as ``1e-3 * min(sigma0)``, and no default
    maximum step size is set. Minimum and maximum step sizes can be changed
    with :meth:`set_min_step_size` and :meth:`set_max_step_size` or through the
    hyper-parameter interface.

    If boundaries are provided, an extra step is added at the end of the
    algorithm that reduces the step size where boundary constraints are not
    met. For :class:`RectangularBoundaries` this works on a per-dimension
    basis::

        while p_j[i] < lower or p_j[i] >= upper:
            step_size_j[i] *= 0.5
            p_j[i] = p_j[i] - sign(df_j[i]) * step_size_j[i]

    For general boundaries a more complex heuristic is used: First, the step
    size in all dimensions is reduced until the boundary constraints are met.
    Next, this reduction is undone for each dimension in turn: if the
    constraint is still met without the reduction the step size in this
    dimension is left unchanged.

    The numbers 0.5 and 1.2 shown in the (main and boundary) pseudo-code are
    technically hyper-parameters, but are fixed in this implementation.

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
        self._f_best = np.inf

        # Minimum and maximum step sizes
        self._step_min = 1e-3 * np.min(self._sigma0)
        self._step_max = None

        # Current point, score, and gradient
        self._current = self._x0
        self._current_f = np.inf
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

        # Rectangular boundaries
        self._lower = self._upper = None
        if isinstance(self._boundaries, pints.RectangularBoundaries):
            self._lower = self._boundaries.lower()
            self._upper = self._boundaries.upper()

        # Reduced step sizes due to boundary violations
        self._breaches = []

    def ask(self):
        """ See :meth:`Optimiser.ask()`. """

        # First call
        if not self._running:
            if (self._step_min is not None and self._step_max is not None
                    and self._step_min >= self._step_max):
                raise Exception(
                    'Max step size must be larger than min step size (current'
                    ' settings: min_step_size = ' + str(self._step_min) + ', '
                    ' max_step_size = ' + str(self._step_max) + ').')
            self._running = True

        # Ready for tell now
        self._ready_for_tell = True

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
        logger.add_string('Bound corr.', 11)

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        logger.log(np.min(self._step_size))
        logger.log(np.max(self._step_size))
        logger.log(','.join([str(x) for x in self._breaches]))

    def max_step_size(self):
        """ Returns the maximum step size (or ``None`` if not set). """
        return self._step_max

    def min_step_size(self):
        """ Returns the minimum step size (or ``None`` if not set). """
        return self._step_min

    def name(self):
        """ See :meth:`Optimiser.name()`. """
        return 'iRprop-'

    def needs_sensitivities(self):
        """ See :meth:`Optimiser.needs_sensitivities()`. """
        return True

    def n_hyper_parameters(self):
        """ See :meth:`pints.TunableMethod.n_hyper_parameters()`. """
        return 2

    def running(self):
        """ See :meth:`Optimiser.running()`. """
        return self._running

    def set_hyper_parameters(self, x):
        """
        See :meth:`pints.TunableMethod.set_hyper_parameters()`.

        The hyper-parameter vector is ``[min_step_size, max_step_size]``.
        """
        self.set_min_step_size(x[0])
        self.set_max_step_size(x[1])

    def set_max_step_size(self, step_size):
        """
        Sets the maximum step size (use ``None`` to let step sizes grow
        indefinitely).
        """
        self._step_max = None if step_size is None else float(step_size)

    def set_min_step_size(self, step_size):
        """
        Sets the minimum step size (use ``None`` to let step sizes shrink
        indefinitely).
        """
        self._step_min = None if step_size is None else float(step_size)

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
            self._current = self._proposed
            self._current_f = fx
            self._current_df = dfx

        # Get product of new and previous gradient
        dprod = dfx * self._current_df

        # Adapt step sizes
        self._step_size[dprod > 0] *= self._eta_max
        self._step_size[dprod < 0] *= self._eta_min

        # Bound step sizes
        if self._step_min is not None:
            self._step_size = np.maximum(self._step_size, self._step_min)
        if self._step_max is not None:
            self._step_size = np.minimum(self._step_size, self._step_max)

        # Remove "weight backtracking"
        # This step ensures that, for each i where dprod < 0:
        #  1. p[i] will remain unchanged this iteration (sign(0) == 0)
        #  2. p[i] will change in the next iteration (dprod == 0), using
        #     the step size set in the current iteration
        dfx[dprod < 0] = 0

        # Update current position
        self._current = self._proposed
        self._current_f = fx
        self._current_df = dfx

        # Take step in direction indicated by current gradient
        p = self._current - self._step_size * np.sign(dfx)

        # Allow boundaries to reduce step size
        if self._lower is not None:
            # Rectangular boundaries: reduce individual step sizes until OK
            out = np.logical_or(p < self._lower, p >= self._upper)
            self._breaches = np.flatnonzero(out)
            sign = np.sign(dfx)
            while np.any(out):
                self._step_size[out] *= self._eta_min
                p = self._current - self._step_size * sign
                out = np.logical_or(p < self._lower, p >= self._upper)

        elif self._boundaries is not None and not self._boundaries.check(p):
            # General boundaries: reduce all step sizes until OK
            s = np.copy(self._step_size)
            sign = np.sign(dfx)
            while not self._boundaries.check(p):
                s *= self._eta_min
                p = self._current - s * sign

            # Attempt restoring one-by-one
            self._breaches = []
            for i, s_big in enumerate(self._step_size):
                small = s[i]
                s[i] = s_big
                if not self._boundaries.check(self._current - s * sign):
                    s[i] = small
                    self._breaches.append(i)
            self._step_size = s
            p = self._current - s * sign

            # An alternative method would be to reduce each dimension's step
            # size one at a time, and check if that restores the boundaries.
            # However, if that doesn't work we then need to test all tuples of
            # dimenions, then triples, etc. The method above looks clumsy, but
            # avoids these combinatorics.

        elif self._breaches:
            # No boundary breaces: empty list (if needed)
            self._breaches = []

        # Store proposed as read-only, so that it can be passed to user
        self._proposed = p
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

