# -*- coding: utf-8 -*-
#
# Slice Sampling with Stepout MCMC Method
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import numpy as np


class SliceStepoutMCMC(pints.SingleChainMCMC):
    r"""
    Implements Slice Sampling with Stepout, as described in [1]_.

    This is a univariate method, which is applied in a
    Slice-Sampling-within-Gibbs framework to allow MCMC sampling from
    multivariate models.

    Generates samples by sampling uniformly from the volume underneath the
    posterior (``f``). It does so by introducing an auxiliary variable (``y``)
    and by definying a Markov chain.

    If the distribution is univariate, sampling follows:

    1. Calculate the PDF (:math:`f(x0)`) of the current sample (:math:`x0`).
    2. Draw a real value (:math:`y`) uniformly from :math`(0, f(x0))`, defining
       a horizontal 'slice' :math:`S = {x: y < f (x)}`. Note that :math:`x0`
       is always within :math:`S`.
    3. Find an interval (:math:`I = (L, R)`) around :math:`x0` that contains
       all, or much, of the slice.
    4. Draw a new point (:math:`x1`) from the part of the slice within this
       interval.

    If the distribution is multivariate, we apply the univariate algorithm to
    each variable in turn, where the other variables are set at their
    current values.

    This implementation uses the "Stepout" method to estimate the interval
    :math:`I = (L, R)`, as described in [1] Fig. 3. pp.715 and consists of the
    following steps:

    1. :math:`U \sim uniform(0, 1)`
    2. :math:`L = x_0 - wU`
    3. :math:`R = L + w`
    4. :math:`V \sim uniform(0, 1)`
    5. :math:`J = floor(mV)`
    6. :math:`K = (m - 1) - J`
    7. while :math:`J > 0` and :math:`y < f(L), L = L - w, J = J - 1`
    8. while :math:`K > 0` and :math:`y < f(R), R = R + w, K = K - 1`

    Intuitively, the interval ``I`` is estimated by expanding the initial
    interval by a width ``w`` in each direction until both edges fall outside
    the slice, or until a pre-determined limit is reached. The parameters
    ``m`` (an integer, which determines the limit of slice size) and
    ``w`` (the estimate of typical slice width) are hyperparameters.

    To sample from the interval :math:`I = (L, R)`, such that the sample
    ``x`` satisfies :math:`y < f(x)`, we use the "Shrinkage" procedure, which
    reduces the size of the interval after rejecting a trial point,
    as defined in [1] Fig. 5. pp.716. This algorithm consists of the
    following steps:

    1. :math:`\bar{L} = L` and :math:`\bar{R} = R`
    2. Repeat:
        a. :math:`U \sim uniform(0, 1)`
        b. :math:`x_1 = \bar{L} + U (\bar{R} - \bar{L})`
        c. if :math:`y < f(x_1)` accept :math:`x_1` and exit loop,
           else:
           if :math:`x_1 < x_0`, :math:`\bar{L} = x_1`
           else :math:`\bar{R} = x_1`

    Intuitively, we uniformly sample a trial point from the interval ``I``,
    and subsequently shrink the interval each time a trial point is rejected.

    The following implementation includes the possibility of carrying out
    "overrelaxed" slice sampling steps, as described in [1] pp. 726.
    Overrelaxed steps increase sampling efficiency in highly correlated
    unimodal distributions by suppressing the random walk behaviour of
    single-variable slice sampling: each variable is still updated in turn,
    but rather than drawing a new value for a variable from its conditional
    distribution independently of the current value, the new value is instead
    chosen to be on the opposite side of the mode from the current value. The
    interval ``I`` is still calculated via Stepout, and the edges ``l,r`` are
    used to estimate the slice endpoints via bisection. To obtain a full
    sampling scheme, overrelaxed updates are alternated with normal Stepout
    updates. To obtain the full benefits of overrelaxation, [1] suggests to
    set almost every update to being overrelaxed and to set the limit ``m``
    for finding ``I`` to infinity. The algorithm consists of the following
    steps:

    1. :math:`\bar{L} = L, \bar{R} = R, \bar{w} = w, \bar{a} = a`
    2. while :math:`R - L < 1.1 * w`:
        a. :math:`M = (\bar{L} + \bar{R})/ 2`
        b. if :math:`\bar{a} = 0 ` or :math:`y < f(M)`, exit loop
        c. if :math:`x_0 > M`, :math:`\bar{L} = M`
           else, :math:`\bar{R} = M`
        d. :math:`\bar{a} = \bar{a} - 1`
        e. :math:`\bar{w} = \bar{w} / 2`
    3. :math:`\hat{L} = \bar{L}, \hat{R} = \bar{R}`
    4. while :math:`\bar{a} > 0`:
        a. :math:`\bar{a} = \bar{a} - 1`
        b. :math:`\bar{w} = \bar{w} \ 2`
        c. if :math:`y >= f(\hat{L} + \bar{w})`, then
           :math:`\hat{L} = \hat{L} + \bar{w}`
        d. if :math:`y >= f(\hat{R} - \bar{w})`, then
           :math:`\hat{R} = \hat{R} - \bar{W}`
    5. :math:`x_1 = \hat{L} + \hat{R} - x_0`
    6. if :math:`x_1 < \bar{L}` or :math:`x_1 >= \bar{R}`
       or :math:`y >= f(x_1)`, then :math:`x_1 = x_0`

    The probability of pursuing an overrelaxed step and the number of bisection
    iterations are hyperparameters.

    To avoid floating-point underflow, we implement the suggestion advanced
    in [1]_ pp.712. We use the log pdf of the un-normalised posterior
    (:math:`g(x) = log(f(x))`) instead of :math:`f(x)`. In doing so, we use an
    auxiliary variable :math:`z = log(y) = g(x0) - \epsilon`, where
    :math:`\epsilon \sim \text{exp}(1)` and define the slice as
    :math:`S = {x : z < g(x)}`.

    Extends :class:`SingleChainMCMC`.

    References
    ----------
    .. [1] Neal, R.M., 2003. "Slice sampling". The annals of statistics, 31(3),
           pp.705-767.
           https://doi.org/10.1214/aos/1056562461
    """

    def __init__(self, x0, sigma0=None):
        super(SliceStepoutMCMC, self).__init__(x0, sigma0)

        # Set initial state
        self._x0 = np.asarray(x0, dtype=float)
        self._running = False
        self._ready_for_tell = False
        self._current = None
        self._current_log_y = None

        self._temporary_log_pdf = None
        self._proposed = None
        self._overrelaxed_step = False

        # Default initial interval width ``w`` used in the Stepout procedure
        # to expand the interval
        self._w = np.abs(self._x0)
        self._w[self._w == 0] = 1
        self._w = 0.1 * self._w

        # Default integer limiting the size of the interval to ``m * w```
        self._m = 50

        # Flag to initialise the expansion of the interval ``I=(L,R)``
        self._first_expansion = False

        # Flag indicating whether the interval expansion is concluded
        self._interval_found = False

        # Number of steps used for expanding the interval ``I``
        self._j = None
        self._k = None

        # Flags used to calculate log_pdf of initial interval edges ``l,r```
        self._init_left = False
        self._init_right = False

        # Edges of the interval ``I``
        self._l = None
        self._r = None

        # Parameter values at interval edges
        self._temp_l = None
        self._temp_r = None

        # Log_pdf of interval edges
        self._fx_l = None
        self._fx_r = None

        # Flags to indicate the interval edge to update
        self._set_l = False
        self._set_r = False

        # Index of parameter ``xi``` we are updating of the sample
        # ``x = (x1,...,xn)``
        self._active_param_index = 0

        # Probability of overrelaxed step
        self._prob_overrelaxed = 0

        # Interval edges used in overrelaxed step
        self._l_bar = None
        self._r_bar = None
        self._l_hat = None
        self._r_hat = None
        self._l_bisection = None
        self._r_bisection = None
        self._temp_l_bisection = None
        self._temp_r_bisection = None
        self._set_l_bisection = False
        self._set_r_bisection = False

        # Integer limiting overrelaxation endpoint accuracy to ``2^(-a) * w``
        self._a = 10

        # Interval width ``w_bar`` used in the overrelaxation step
        self._w_bar = None

        # Mid-point of overrelaxed interval
        self._mid = None
        self._temp_mid = None
        self._fx_mid = None

        # Flags used for overrelaxed step
        self._init_overrelaxation = False
        self._init_narrowing = False
        self._init_bisection = False
        self._bisection = False

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """

        # Check ask/tell pattern
        if self._ready_for_tell:
            raise RuntimeError('Ask() called when expecting call to tell().')

        # Initialise on first call
        if not self._running:
            self._running = True

        # Very first iteration
        if self._current is None:

            # Ask for the log pdf of x0
            self._ready_for_tell = True
            return np.array(self._x0, copy=True)

        # Initialise the expansion of interval ``I=(l,r)``
        if self._first_expansion:

            # Set initial values for l and r
            u = np.random.uniform()
            self._l = (self._proposed[self._active_param_index] -
                       self._w[self._active_param_index] * u)
            self._r = self._l + self._w[self._active_param_index]

            # Set maximum number of steps for expansion to the left (j)
            # and right (k)
            v = np.random.uniform()
            self._j = np.floor(self._m * v)
            self._k = (self._m - 1) - self._j

            # Initialise arrays used for calculating the log_pdf of the edges
            self._temp_l = np.array(self._proposed, copy=True)
            self._temp_r = np.array(self._proposed, copy=True)
            self._temp_l[self._active_param_index] = self._l
            self._temp_r[self._active_param_index] = self._r

            # Set flags to calculate log_pdf of ``l,r``
            self._init_left = True
            self._init_right = True

            self._first_expansion = False

        # Ask for log_pdf of initial edges ``l,r```
        if self._init_left:
            self._ready_for_tell = True
            return np.array(self._temp_l, copy=True)

        if self._init_right:
            self._ready_for_tell = True
            return np.array(self._temp_r, copy=True)

        # Expand the interval ``I``` until edges ``l,r`` are outside the slice
        # or we have reached limit of expansion steps

        # Check whether we can expand to the left
        if self._j > 0 and self._current_log_y < self._fx_l:

            # Set flag to indicate that we are updating the left edge
            self._set_l = True

            # Expand interval to the left
            self._l -= self._w[self._active_param_index]
            self._temp_l[self._active_param_index] = self._l
            self._j -= 1

            # Ask for log pdf of the updated left edge
            self._ready_for_tell = True
            return np.array(self._temp_l, copy=True)

        # Reset flag now that we have finished updating the left edge
        self._set_l = False

        # Check whether we can expand to the right
        if self._k > 0 and self._current_log_y < self._fx_r:

            # Set flag to indicate that we are updating the right edge
            self._set_r = True

            # Expand interval to the right
            self._r += self._w[self._active_param_index]
            self._temp_r[self._active_param_index] = self._r
            self._k -= 1

            # Ask for log pdf of the updated right edge
            self._ready_for_tell = True
            return np.array(self._temp_r, copy=True)

        # Reset flag now that we have finished updating the right edge
        self._set_r = False

        # Now that we have expanded the interval, set flag
        self._interval_found = True

        # Overrelaxed step
        if self._overrelaxed_step:

            # Initialise variables for overrelaxed step
            if self._init_overrelaxation:
                self._l_bar = self._l
                self._r_bar = self._r
                self._w_bar = self._w[self._active_param_index]
                self._a_bar = self._a
                self._temp_mid = np.array(self._proposed, copy=True)
                self._init_overrelaxation = False
                self._init_narrowing = True
                self._init_bisection = True

            # If interval is of size ``w``, narrow it until mid-point is
            # within the slice
            if (((self._r - self._l) < 1.1 * self._w[self._active_param_index])
                    and self._init_narrowing):

                # Ask for log pdf of interval mid point
                self._mid = (self._l_bar + self._r_bar) / 2
                self._temp_mid[self._active_param_index] = self._mid
                self._ready_for_tell = True
                return np.array(self._temp_mid, copy=True)

            # Initialise endpoints for bisection
            if self._init_bisection:
                self._l_hat = self._l_bar
                self._r_hat = self._r_bar
                self._init_bisection = False

            # Apply bisection to endpoint edges
            if self._a_bar > 0:

                # Prepare bisection
                if self._bisection:
                    self._w_bar = self._w_bar / 2
                    self._temp_l_bisection = np.array(self._proposed,
                                                      copy=True)
                    self._temp_r_bisection = np.array(self._proposed,
                                                      copy=True)
                    self._set_l_bisection = True
                    self._set_r_bisection = True
                    self._bisection = False

                # Apply bisection to left edge
                if self._set_l_bisection:

                    self._l_bisection = (self._l_hat + self._w_bar)
                    self._temp_l_bisection[self._active_param_index] = (
                        self._l_bisection)
                    self._ready_for_tell = True
                    return np.array(self._temp_l_bisection, copy=True)

                # Apply bisection to right edge
                if self._set_r_bisection:
                    self._r_bisection = (self._r_hat - self._w_bar)
                    self._temp_r_bisection[self._active_param_index] = (
                        self._r_bisection)
                    self._ready_for_tell = True
                    return np.array(self._temp_r_bisection, copy=True)

            # Find candidate point by flipping from the current point to
            # the opposide side
            self._proposed[self._active_param_index] = (
                self._l_hat + self._r_hat
                - self._current[self._active_param_index])
            self._ready_for_tell = True
            return np.array(self._proposed, copy=True)

        else:
            # Sample new trial point by sampling uniformly from the
            # interval ``I=(l,r)``
            u = np.random.uniform()
            self._proposed[self._active_param_index] = \
                self._l + u * (self._r - self._l)

            # Send trial point for checks
            self._ready_for_tell = True
            return np.array(self._proposed, copy=True)

    def bisection_steps(self):
        """
        Returns integer limit overrelaxation endpoint accuracy to
        ``2^(-bisection steps) * width``.
        """
        return self._a

    def current_slice_height(self):
        """
        Returns current height value used to define the current slice.
        """
        return self._current_log_y

    def expansion_steps(self):
        """
        Returns integer used for limiting interval expansion.
        """
        return self._m

    def prob_overrelaxed(self):
        """
        Returns probability of carrying out an overrelaxed step.
        """
        return self._prob_overrelaxed

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Slice Sampling - Stepout'

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 4

    def set_bisection_steps(self, a):
        """
        Set integer for limiting the bisection process in overrelaxed steps.
        """
        a = int(a)
        if a < 0:
            raise ValueError(
                'Integer must be positive (to limit overrelaxation endpoint'
                ' accuracy to (2 ^ (-bisection steps) * width).')
        self._a = a

    def set_expansion_steps(self, m):
        """
        Set integer for limiting the interval expansion.
        """
        m = int(m)
        if m <= 0:
            raise ValueError('Integer must be positive to limit the'
                             ' interval size to ``integer * width``.')
        self._m = m

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[width, expansion steps,
        prob_overrelaxed, bisection steps]``.
        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_width(x[0])
        self.set_expansion_steps(x[1])
        self.set_prob_overrelaxed(x[2])
        self.set_bisection_steps(x[3])

    def set_prob_overrelaxed(self, prob):
        """
        Set the probability of a step being overrelaxed.
        """
        prob = float(prob)
        if prob < 0 or prob > 1:
            raise ValueError('Probability must be positive and <= 1.')
        self._prob_overrelaxed = prob

    def set_width(self, w):
        """
        Sets the width for generating the interval.

        This can either be a single number or an array with the same number of
        elements as the number of variables to update.
        """
        if np.isscalar(w):
            w = np.ones(self._n_parameters) * w
        else:
            w = np.array(w, copy=True)
            if len(w) != self._n_parameters:
                raise ValueError(
                    'Width for interval expansion must a scalar or an array'
                    ' of length n_parameters.')
        if np.any(w < 0):
            raise ValueError('Width for interval expansion must be positive.')
        self._w = w

    def tell(self, fx):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """

        # Check ask/tell pattern
        if not self._ready_for_tell:
            raise RuntimeError('Tell called before proposal was set.')
        self._ready_for_tell = False

        # Ensure fx is a float
        fx = float(fx)

        # Very first call
        if self._current is None:

            # Check first point is somewhere sensible
            if not np.isfinite(fx):
                raise ValueError(
                    'Initial point for MCMC must have finite logpdf.')

            # Set current sample, log pdf of current sample and initialise
            # proposed sample for next iteration
            self._current = np.copy(self._x0)
            self._temporary_log_pdf = fx
            self._proposed = np.copy(self._current)

            # Sample height of the slice log_y for next iteration
            e = np.random.exponential(1)
            self._current_log_y = fx - e

            # Set flag to true as we need to initialise the interval expansion
            # for next iteration
            self._first_expansion = True

            # Check whether next mcmc step should be overrelaxed
            self._overrelaxed_step = (np.random.uniform() <
                                      self._prob_overrelaxed)
            if self._overrelaxed_step:
                self._init_overrelaxation = True
                self._bisection = True

            # Return first point in chain, which is x0
            return np.copy(self._current), fx, True

        # While we expand the interval ``I=(l,r)``, we return None
        if not self._interval_found:

            # Set the log_pdf of the interval edge that we are expanding
            if self._set_l:
                self._fx_l = fx
            elif self._set_r:
                self._fx_r = fx
            elif self._init_left:
                self._fx_l = fx
                self._init_left = False
            elif self._init_right:
                self._fx_r = fx
                self._init_right = False
            return None

        # Overrelaxed step
        if self._overrelaxed_step:

            # When the interval is of size ``w``, narrow until mid-point
            # is inside the slice
            if (((self._r - self._l) < 1.1 *
                    self._w[self._active_param_index]) and
                    self._init_narrowing):

                self._fx_mid = fx

                # Once the mid-point is within the slice or narrowing limit is
                # reached, break narrowing loop
                if (self._a_bar == 0 or (self._current_log_y <
                                         self._fx_mid)):
                    self._init_narrowing = False
                    return None

                # Narrow interval
                if (self._current[self._active_param_index] >
                        self._temp_mid[self._active_param_index]):
                    self._l_bar = self._mid
                else:
                    self._r_bar = self._mid
                self._a_bar -= 1
                self._w_bar = self._w_bar / 2
                return None

            # Apply bisection to left edge
            if self._set_l_bisection:
                if self._current_log_y >= fx:
                    self._l_hat = (self._l_hat + self._w_bar)
                self._set_l_bisection = False
                return None

            # Apply bisection to right edge
            if self._set_r_bisection:
                if self._current_log_y >= fx:
                    self._r_hat = (self._r_hat - self._w_bar)
                self._set_r_bisection = False

                # Reset flag for next bisection iteration
                self._bisection = True

                # Decrease count of bisection steps left
                self._a_bar -= 1
                return None

            # If trial point is not acceptable, maintain current state
            if (self._proposed[self._active_param_index] < self._l_bar or
                    self._proposed[self._active_param_index] > self._r_bar or
                    self._current_log_y >= fx):

                # Reset proposal to undo last change
                self._proposed[self._active_param_index] = (
                    self._current[self._active_param_index])

                # And update fx to the corresponding log pdf (needed below!)
                fx = self._temporary_log_pdf

            # Reset flags for next interval expansion
            self._first_expansion = True
            self._interval_found = False

            # Reset overrelaxation flags
            self._init_overrelaxation = True
            self._bisection = True

            # Reset active parameter indices
            if self._active_param_index == len(self._proposed) - 1:
                self._active_param_index = 0

                # The accepted sample becomes the new current sample
                self._current = np.copy(self._proposed)

                # The log_pdf of the accepted sample is used to construct the
                # new slice
                self._temporary_log_pdf = fx

                # Sample new log_y used to define the next slice
                e = np.random.exponential(1)
                self._current_log_y = fx - e

                # Check whether next mcmc step should be overrelaxed
                self._overrelaxed_step = (np.random.uniform() <
                                          self._prob_overrelaxed)
                return np.copy(self._current), fx, True

            else:
                self._temporary_log_pdf = fx
                self._active_param_index += 1
                return None

        # Normal Stepout step
        else:
            # Do ``Threshold Check`` to check if the proposed point is within
            # the slice
            if self._current_log_y < fx:

                self._first_expansion = True
                self._interval_found = False

                # Reset active parameter indices
                if self._active_param_index == len(self._proposed) - 1:

                    self._active_param_index = 0

                    # The accepted sample becomes the new current sample
                    self._current = np.copy(self._proposed)

                    # The log_pdf of the accepted sample is used to construct
                    # the new slice
                    self._temporary_log_pdf = fx

                    # Sample new log_y used to define the next slice
                    e = np.random.exponential(1)
                    self._current_log_y = fx - e

                    # Check whether next mcmc step should be overrelaxed
                    self._overrelaxed_step = (np.random.uniform() <
                                              self._prob_overrelaxed)
                    if self._overrelaxed_step:
                        self._init_overrelaxation = True
                        self._bisection = True
                    return np.copy(self._current), fx, True

                else:
                    self._temporary_log_pdf = fx
                    self._active_param_index += 1
                    return None

            # If the trial point is rejected in the ``Threshold Check``, shrink
            # the interval
            if (self._proposed[self._active_param_index] <
                    self._current[self._active_param_index]):
                self._l = self._proposed[self._active_param_index]
                self._temp_l[self._active_param_index] = self._l
            else:
                self._r = self._proposed[self._active_param_index]
                self._temp_r[self._active_param_index] = self._r
            return None

    def width(self):
        """
        Returns the width used for generating the interval.
        """
        return np.copy(self._w)
