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
    *Extends:* :class:`SingleChainMCMC`

    Implements Hyperrectangles-based Multivariate Slice Sampling,
    as described in [1].
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
        self._adaptive = True

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

        # Sample ith parameter for new proposal
        for i in range(self._n_parameters):
            u = np.random.uniform()
            self._proposed[i] = (self._L[i] + u * (self._R[i] - self._L[i]))

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
        fx, grad = reply

        # Check reply, copy gradient
        fx = float(fx)
        grad = pints.vector(grad)
        assert(grad.shape == (self._n_parameters, ))

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

            # Sample new log_y used to define the next slice
            e = np.random.exponential(1)
            self._current_log_y = fx - e

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
                for i, x_1 in enumerate(self._proposed):
                    if x_1 < self._current[i]:
                        self._L[i] = x_1
                    else:
                        self._R[i] = x_1

    def needs_sensitivities(self):
        """ See :meth:`pints.MCMCSampler.needs_sensitivities()`. """
        return True
        
    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Slice Sampling - Hyperrectangles'

    def set_w(self, w):
        """
        Sets scale vector "w" for generating the hyperrectangle.
        """
        if type(w) == int or float:
            w = np.full((len(self._x0)), w)
        else:
            w = np.asarray(w)
        if any(n < 0 for n in w):
            raise ValueError("""Width "w" must be positive for
                            interval expansion.""")
        self._w = w

    def set_adaptive_shrinking(self, adaptive):
        """
        Sets adaptive method for shrinking the hyperrectangle.
        """
        if type(adaptive) != bool:
            raise ValueError("""Variable should be either "True"
                              or "False" to set adaptive shrinking
                              method""")
        self._adaptive = adaptive

    def get_w(self):
        """
        Returns scale w used for generating the hyperrectangle.
        """
        return self._w

    def get_adaptive_shrinking(self):
        """
        Returns True/False if adaptive shrinking is on/off.
        """
        return self._adaptive

    def get_current_slice_height(self):
        """
        Returns current log_y used to define the current slice.
        """
        return self._current_log_y

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 2

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[w, adaptive]``.
        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_w(x[0])
        self.set_adaptive_shrinking(x[1])