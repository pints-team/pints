#
# Slice Sampling with Stepout MCMC Method
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class SliceStepoutMCMC(pints.SingleChainMCMC):
    """
    *Extends:* :class:`SingleChainMCMC`

    Implements Slice Sampling with Stepout, as described in [1]. This is a
    univariate method, which is applied in a Slice-Sampling-within-Gibbs
    framework to allow MCMC sampling from multivariate models.

    Generates samples by sampling uniformly from the volume underneath the
    posterior (``f``). It does so by introducing an auxiliary variable (``y``)
    and by definying a Markov chain.

    If the distribution is univariate, sampling follows:

    1) Calculate the pdf (``f(x0)``) of the current sample (``x0``).
    2) Draw a real value (``y``) uniformly from (0, f(x0)), defining a
    horizontal “slice”: S = {x: y < f (x)}. Note that ``x0`` is
    always within S.
    3) Find an interval (``I = (L, R)``) around ``x0`` that contains all,
    or much, of the slice.
    4) Draw a new point (``x1``) from the part of the slice
    within this interval.

    If the distribution is multivariate, we apply the univariate algorithm to
    each variable in turn, where the other variables are set at their
    current values.

    This implementation uses the ``Stepout`` method to estimate the interval
    ``I = (L, R)``, as described in [1] Fig. 3. pp.715 and consists of the
    following steps:

    1. ``U \sim uniform(0, 1)``
    2. ``L = x_0 - wU``
    3. ``R = L + w``
    4. ``V \sim uniform(0, 1)``
    5. ``J = floor(mV)``
    6. ``K = (m - 1) - J``
    6. while ``J > 0`` and ``y < f(L)``, ``L = L - w`` and ``J = J - 1``
    7. while ``K > 0`` and ``y < f(R)``, ``R = R + w`` and ``K = K - 1``

    Intuitively, the interval ``I`` is estimated by expanding the initial
    interval by a width ``w`` in each direction until both edges fall outside
    the slice, or until a pre-determined limit is reached. The parameters
    ``m`` (an integer, which determines the limit of slice size) and
    ``w`` (the estimate of typical slice width) are hyperparameters.

    To sample from the interval ``I = (L, R)``, such that the sample
    ``x`` satisfies ``y < f(x)``, we use the ``Shrinkage`` procedure, which
    reduces the size of the interval after rejecting a trial point,
    as defined in [1] Fig. 5. pp.716. This algorithm consists of the
    following steps:

    1. ``\bar{L} = L`` and ``\bar{R} = R``
    2. Repeat:
        a. ``U \sim uniform(0, 1)``
        b. ``x_1 = \bar{L} + U (\bar{R} - \bar{L})``
        c. if ``y < f(x_1)`` accept ``x_1`` and exit loop,
           else
            if ``x_1 < x_0``, ``\bar{L} = x_1``
            else ``\bar{R} = x_1``

    Intuitively, we uniformly sample a trial point from the interval ``I``,
    and subsequently shrink the interval each time a trial point is rejected.

    To avoid floating-point underflow, we implement the suggestion advanced
    in [1] pp.712. We use the log pdf of the un-normalised posterior
    (``g(x) = log(f(x))``) instead of ``f(x)``. In doing so, we use an
    auxiliary variable ``z = log(y) = g(x0) − \epsilon``, where
    ``\epsilon \sim \text{exp}(1)`` and define the slice as
    S = {x : z < g(x)}.

    [1] Neal, R.M., 2003. Slice sampling. The annals of statistics, 31(3),
    pp.705-767.
    """

    def __init__(self, x0, sigma0=None):
        super(SliceStepoutMCMC, self).__init__(x0, sigma0)

        # Set initial state
        self._x0 = np.asarray(x0, dtype=float)
        self._running = False
        self._ready_for_tell = False
        self._current = None
        self._current_log_pdf = None
        self._current_log_y = None
        self._proposed = None

        # Default initial interval width w used in the Stepout procedure
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

        # Number of steps used for expanding the interval ``I=(L,R)``
        self._j = None
        self._k = None

        # Edges of the interval ``I=(L,R)``
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

        # Index of parameter "xi" we are updating of the sample
        # "x = (x1,...,xn)"
        self._active_param_index = 0

        # Flags used to calculate log_pdf of initial interval edges ``l,r```
        self._init_left = False
        self._init_right = False

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

            self._first_expansion = False

            # Set flags to calculate log_pdf of ``l,r``
            self._init_left = True
            self._init_right = True

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

        # Sample new trial point by sampling uniformly from the
        # interval ``I=(l,r)``
        u = np.random.uniform()
        self._proposed[self._active_param_index] = self._l + u * (self._r -
                                                                  self._l)

        # Send trial point for checks
        self._ready_for_tell = True
        return np.array(self._proposed, copy=True)

    def tell(self, reply):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """

        # Check ask/tell pattern
        if not self._ready_for_tell:
            raise RuntimeError('Tell called before proposal was set.')
        self._ready_for_tell = False

        # Unpack reply
        fx = np.asarray(reply, dtype=float)

        # Very first call
        if self._current is None:

            # Check first point is somewhere sensible
            if not np.isfinite(fx):
                raise ValueError(
                    'Initial point for MCMC must have finite logpdf.')

            # Set current sample, log pdf of current sample and initialise
            # proposed sample for next iteration
            self._current = np.array(self._x0, copy=True)
            self._current_log_pdf = fx
            self._proposed = np.array(self._current, copy=True)

            # Sample height of the slice log_y for next iteration
            e = np.random.exponential(1)
            self._current_log_y = self._current_log_pdf - e

            # Set flag to true as we need to initialise the interval expansion
            # for next iteration
            self._first_expansion = True

            # Return first point in chain, which is x0
            return np.array(self._current, copy=True)

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

        # Do ``Threshold Check`` to check if the proposed point is within
        # the slice
        if self._current_log_y < fx:

            self._first_expansion = True
            self._interval_found = False

            # Reset active parameter indices
            if self._active_param_index == len(self._proposed) - 1:

                self._active_param_index = 0

                # The accepted sample becomes the new current sample
                self._current = np.array(self._proposed, copy=True)

                # The log_pdf of the accepted sample is used to construct the
                # new slice
                self._current_log_pdf = fx

                # Sample new log_y used to define the next slice
                e = np.random.exponential(1)
                self._current_log_y = self._current_log_pdf - e
                return np.array(self._proposed, copy=True)

            else:
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

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Slice Sampling - Stepout'

    def set_w(self, w):
        """
        Sets width w for generating the interval.
        """
        if type(w) == int or float:
            w = np.full((len(self._x0)), w)
        else:
            w = np.asarray(w)
        if any(n < 0 for n in w):
            raise ValueError("""Width w must be positive for
                            interval expansion.""")
        self._w = w

    def set_m(self, m):
        """
        Set integer m for limiting interval expansion.
        """
        m = int(m)
        if m <= 0:
            raise ValueError("""Integer m must be positive to limit the
                            interval size to "m * w".""")
        self._m = m

    def get_w(self):
        """
        Returns width w used for generating the interval.
        """
        return self._w

    def get_m(self):
        """
        Returns integer m used for limiting interval expansion.
        """
        return self._m

    def current_log_pdf(self):
        """ See :meth:`SingleChainMCMC.current_log_pdf()`. """
        return self._current_log_pdf

    def current_slice_height(self):
        """
        Returns current log_y used to define the current slice.
        """
        return self._current_log_y

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 2

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[w, m]``.
        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_w(x[0])
        self.set_m(x[1])
