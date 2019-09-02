# -*- coding: utf-8 -*-
#
# Hyperrectangles-based Slice Sampling
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


class SliceHyperrectanglesMCMC(pints.SingleChainMCMC):
    """
    Implements Hyperrectangles-based Slice Sampling, as described in [1].

    This is a multivariate method, which generates n-dimensional samples of
    the form ``x = (x_1, ..., x_n)`` by sampling uniformly from the area of an
    axis-aligned hyperrectangle:
    ``H = {x: L_i < x_i < R_i for all i = 1, ..., n}``.
    Here, ``L_i`` and ``R_i`` define  the extent of the hyperrectangle along
    the ``i`` th axis.

    Sampling follows:

    1. Calculate the pdf (``f(x0)``) of the current sample (``x0``).
    2. Draw a real value (``y``) uniformly from (0, f(x0)), defining a
       horizontal “slice”: S = {x: y < f (x)}. Note that ``x0`` is
       always within S.
    3. Find a hyperrectangle (``H = (L_1, R_1) ×···× (L_n, R_n)``) around
       ``x_0``, which preferably contains at least a big part of the slice.
    4. Draw a new point (``x1``) from the part of the slice within this
       hyperrectangle.

    The implementation uses estimates (``w_i``) of the relative scales of the
    variables to randomly position a hyperrectangle with such dimensions
    uniformly over positions containing ``x_0`` that lead to ``H``. The
    algorithm consists of the following steps, as described in [1] Fig. 8.
    pp.723:

    1. ``y \sim uniform(0, f(x_0))``
    2. for ``i = 1`` to ``n``:
        a. ``U_i \sim uniform(0,1)``
        b. ``L_i = x_{0_i} - w_i U_i``
        c. ``L_i + w_i``
    3. Repeat:
        a. for ``i = 1`` to ``n``:
            - ``U_i \sim uniform(0,1)``
            - ``x_{1_i} = L_i + U_i (R_i - L_i)``
        b. if ``y < f(x_1)``, exit
        c. for ``i = 1`` to ``n``:
            - if ``x_{1_i} < x_{0_i}``, ``L_i = x_{1_i}``
            - else, ``R_i = x_{1_i}``

    In the presented algorithm, the hyperrectangle is homogeneously shrunk
    in all directions when a proposal is drawn outside the slice, until an
    acceptable sample is found.

    The following implementation includes the option of executing an
    adaptive shrinkage procedure along only one axis. This is determined using
    the gradient and the current dimensions of the hyperrectangle,
    as described in [1] pp. 722. Specifically, only the axis corresponding
    to the variable ``x_i`` is shrunk, where ``i`` maximises:
    ``(R_i - L_i) |G_i|``, with ``G`` being the gradient of ``f(x)` evaluated
    at the last rejected sample. By multiplying the magnitude of the component
    ``i`` of the gradient by the width of the hyperrectangle in this direction,
    we get an estimate of the amount by which log ``f(x)`` changes along axis
    ``i``. The axis for which this change is thought to be largest is likely
    to be the best one to shrink in order to eliminate points outside the
    slice.

    To avoid floating-point underflow, we implement the suggestion advanced
    in [1] pp.712. We use the log pdf of the un-normalised posterior
    (``g(x) = log(f(x))``) instead of ``f(x)``. In doing so, we use an
    auxiliary variable ``z = log(y) = g(x0) − \epsilon``, where
    ``\epsilon \sim \text{exp}(1)`` and define the slice as
    S = {x : z < g(x)}.

    [1] Neal, R.M., 2003. Slice sampling. The annals of statistics, 31(3),
    pp.705-767.

    *Extends:* :class:`SingleChainMCMC`
    """

    def __init__(self, x0, sigma0=None):
        super(SliceHyperrectanglesMCMC, self).__init__(x0, sigma0)

        # Set initial state
        self._x0 = np.asarray(x0, dtype=float)
        self._running = False
        self._ready_for_tell = False
        self._current = None
        self._current_log_y = None
        self._proposed = None
        self._hyperrectangle_positioned = False

        # Hyperrectangle edges
        self._L = np.zeros(len(self._x0))
        self._R = np.zeros(len(self._x0))

        # Default scale estimates for each variable
        self._w = np.abs(self._x0)
        self._w[self._w == 0] = 1
        self._w = 0.1 * self._w

        # Flag to turn on adaptive shrinking
        self._adaptive = False

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

        # Randomly position hyperrectangle:
        # ``H = (L_1, R_1) x ... x (L_n, R_n)``
        if not self._hyperrectangle_positioned:
            for i, w in enumerate(self._w):
                u = np.random.uniform()
                self._L[i] = self._current[i] - w * u
                self._R[i] = self._L[i] + w
            self._hyperrectangle_positioned = True

        # Sample new proposal
        for i in range(self._n_parameters):
            u = np.random.uniform()
            self._proposed[i] = (self._L[i] + u * (self._R[i] - self._L[i]))

        # Send trial point for checks
        self._ready_for_tell = True
        return np.array(self._proposed, copy=True)

    def adaptive_shrinking(self):
        """
        Returns True/False if adaptive shrinking is on/off.
        """
        return self._adaptive

    def current_log_pdf(self):
        """ See :meth:`SingleChainMCMC.current_log_pdf()`. """
        return np.copy(self._current_log_pdf)

    def current_slice_height(self):
        """
        Returns current height value used to define the current slice.
        """
        return self._current_log_y

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Slice Sampling - Hyperrectangles'

    def needs_sensitivities(self):
        """ See :meth:`pints.MCMCSampler.needs_sensitivities()`. """
        return True

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 2

    def set_adaptive_shrinking(self, adaptive):
        """
        Turns on/off the adaptive method for shrinking the hyperrectangle.
        """
        self._adaptive = bool(adaptive)

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[width, adaptive]``.
        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_width(x[0])
        self.set_adaptive_shrinking(x[1])

    def set_width(self, w):
        """
        Sets the width for generating the interval. This can either
        be a single number or an array with the same number of elements
        as the number of variables to update.
        """
        if type(w) == int or float:
            w = np.full((len(self._x0)), w)
        if any(n < 0 for n in w):
            raise ValueError('Width must be positive'
                             'for interval expansion.')
        self._w = w

    def tell(self, reply):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """

        # Check ask/tell pattern
        if not self._ready_for_tell:
            raise RuntimeError('Tell called before proposal was set.')
        self._ready_for_tell = False

        # Unpack reply
        fx, grad = reply
        fx = float(fx)
        grad = pints.vector(grad)

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

            # Return first point in chain, which is x0
            return np.array(self._current, copy=True)

        # Subsequent calls
        if self._current_log_y < fx:
            # The accepted sample becomes the new current sample
            self._current = np.array(self._proposed, copy=True)
            self._current_log_pdf = fx

            # Sample new log_y used to define the next slice
            e = np.random.exponential(1)
            self._current_log_y = self._current_log_pdf - e

            self._hyperrectangle_positioned = False

            # Return accepted sample
            return np.array(self._proposed, copy=True)

        # Shrinking
        else:
            # Adaptive shrinking: shrink in the direction ``index``
            # in which ``(R_i - L_i) |G_i|`` is maximised
            if self._adaptive:
                # Store products ``(R_i - L_i) |G_i|``
                temp = np.zeros(self._n_parameters)
                for i in range(self._n_parameters):
                    temp[i] = (self._R[i] - self._L[i]) * np.abs(grad[i])

                # Index which maximises ``(R_i - L_i) |G_i|``
                index = np.argmax(temp)

                # Shrink only in the direction ``index``
                if self._proposed[index] < self._current[index]:
                    self._L[index] = self._proposed[index]
                else:
                    self._R[index] = self._proposed[index]

            # Shrink homogeneously in all directions
            else:
                for i, x_1i in enumerate(self._proposed):
                    if x_1i < self._current[i]:
                        self._L[i] = x_1i
                    else:
                        self._R[i] = x_1i

    def width(self):
        """
        Returns widths used for generating the hyperrectangle.
        """
        return np.copy(self._w)
