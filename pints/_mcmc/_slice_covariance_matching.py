# -*- coding: utf-8 -*-
#
# Slice Sampling - Covariance Adaptive: Covariance Matching
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


class SliceCovarianceMatchingMCMC(pints.SingleChainMCMC):
    """
    *Extends:* :class:`SingleChainMCMC`
    """

    def __init__(self, x0, sigma0=None):
        super(SliceCovarianceMatchingMCMC, self).__init__(x0, sigma0)

        # Set initial state
        self._x0 = np.asarray(x0, dtype=float)
        self._running = False
        self._ready_for_tell = False
        self._current = None
        self._proposed_pdf = None
        self._current_log_y = None
        self._proposed = None
        self._log_fx_u = None
        self._calculate_fx_u = False
        self._sent_proposal = False

        # Standard deviation of initial crumb
        self._sigma_c = 1

        # Crumb
        self._c = None

        # Estimate of the density at the mode
        self._M = None

        # Initialise un-normalised crumb mean
        self._c_bar_star = np.zeros(self._n_parameters)

        # Define Cholesky upper triangles: F for W_k and R_k for Lambda_k
        self._R = self._sigma_c ** (-1) * np.identity(self._n_parameters)
        self._F = self._sigma_c ** (-1) * np.identity(self._n_parameters)

        # Unnormalised gradient calculated at rejected proposal
        self._G = None

        # Normalised gradient calculated at rejected proposal
        self._g = None

        # Distance between rejected proposal and crumb
        self._delta = None

        # Parameter to control variance precision
        self._theta = 1

    # Define function for Cholesky rank one update
    def _chud(self, matrix, vector):
        V = np.dot(matrix.transpose(), matrix) + np.outer(vector, vector)
        chol = np.linalg.cholesky(V).transpose()
        return np.array(chol, copy=True)

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

        # If the proposal has been rejected, we need to evaluate ``f(u)``
        # in order to find the covariance matrix for the next proposal
        if self._calculate_fx_u:
            # Ask for the log pdf of ``u``
            self._ready_for_tell = True
            self._calculate_fx_u = False
            return np.array(self._u, copy=True)

        # Draw first p-variate
        mean = np.zeros(self._n_parameters)
        cov = np.identity(self._n_parameters)
        z = np.random.multivariate_normal(mean, cov)

        # Draw crumb c
        self._c = self._current + np.dot(np.linalg.inv(self._F), z)

        # Compute un-normalised proposal mean
        self._c_bar_star = self._c_bar_star + np.dot(np.dot(np.transpose(
            self._F), self._F), self._c)

        # Compute normalised proposal mean
        c_bar = np.dot(np.dot(np.linalg.inv(self._R), np.transpose(
            np.linalg.inv(self._R))), self._c_bar_star)

        # Draw second p-variate
        z = np.random.multivariate_normal(mean, cov)

        # Draw sample: x
        self._proposed = c_bar + np.dot(np.linalg.inv(self._R), z)

        # Set flag indicating we have created a new proposal. This is used to
        # store the value of the log_pdf of the proposed point in the tell()
        # method
        self._sent_proposal = True

        # Set flag that will be used if the proposal is rejected
        self._calculate_fx_u = True

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

        # If this is the log_pdf of a new point, save the value and use it
        # to check ``f(x_1) >= y``
        if self._sent_proposal:
            self._proposed_pdf = fx
            self._sent_proposal = False

        # Very first call
        if self._current is None:
            # Check first point is somewhere sensible
            if not np.isfinite(fx):
                raise ValueError(
                    'Initial point for MCMC must have finite logpdf.')

            # Set current sample, log pdf of current sample and initialise
            # proposed sample for next iteration
            self._current = np.array(self._x0, copy=True)
            self._proposed = np.array(self._current, copy=True)
            self._current_log_pdf = fx
            self._M = fx

            # Sample height of the slice log_y for next iteration
            e = np.random.exponential(1)
            self._current_log_y = self._M - e

            # Return first point in chain, which is x0
            return np.array(self._current, copy=True)

        # Acceptance check
        if self._proposed_pdf >= self._current_log_y:
            # The accepted sample becomes the new current sample
            self._current = np.array(self._proposed, copy=True)

            # Update estimate of mode density
            self._M = fx

            # Sample new log_y used to define the next slice
            e = np.random.exponential(1)
            self._current_log_y = self._M - e

            # Reset parameters
            self._c_bar_star = np.zeros(self._n_parameters)
            self._F = self._sigma_c ** (-1) * np.identity(self._n_parameters)
            self._R = self._sigma_c ** (-1) * np.identity(self._n_parameters)
            self._calculate_fx_u = False

            # Return accepted sample
            return np.array(self._proposed, copy=True)

        # If proposal is rejected, we define new covariance matrix for
        # next proposal distribution
        else:
            if self._calculate_fx_u:
                # Gradient evaluated at rejected proposal
                self._G = grad

                # Calculate normalised gradient
                self._g = self._G / np.linalg.norm(self._G)

                # Calculate distance between proposal and crumb
                self._delta = np.linalg.norm(self._proposed - self._c)

                # Generate new point ``u``
                self._u = self._proposed + self._delta * self._g
                return None

            else:
                self._log_fx_u = fx

                # Calculate kappa
                kappa = (-2.) * self._delta ** (-2.) * (
                    self._log_fx_u - self._proposed_pdf - self._delta *
                    np.linalg.norm(self._G))

                # Peak of parabolic cut through ``x1`` and ``u``
                lxu = (0.5 * (np.linalg.norm(self._G) ** 2 / kappa) +
                       self._proposed_pdf)

                # Update M
                self._M = max(self._M, lxu)

                # Calculate conditional variance of new distribution
                sigma_squared = 2 / 3 * (self._M - self._current_log_y) / kappa

                alpha = max(
                    0, sigma_squared ** (-1) - ((
                        1 + self._theta) * np.dot(
                            np.transpose(self._g), np.dot(
                                np.transpose(self._R), np.dot(
                                    self._R, self._g)))))

                # Update F and R
                self._F = self._chud(
                    np.sqrt(self._theta) * self._R, np.sqrt(alpha) * self._g)
                self._R = self._chud(
                    np.sqrt(1 + self._theta) * self._R, np.sqrt(alpha) *
                    self._g)

                return None

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Slice Sampling - Covariance Adaptive: Covariance Matching'

    def needs_sensitivities(self):
        """ See :meth:`pints.MCMCSampler.needs_sensitivities()`. """
        return True

    def set_sigma_c(self, sigma_c):
        """
        Sets standard deviation of initial crumb.
        """
        sigma_c = float(sigma_c)
        if sigma_c < 0:
            raise ValueError("""Inital crumb standard deviationn
                             must be positive.""")
        self._sigma_c = sigma_c

    def set_theta(self, theta):
        """
        Sets parameter to control how fast the precision of the
        distribution increases.
        """
        self._theta = float(theta)

    def get_sigma_c(self):
        """
        Returns standard deviation of initial crumb.
        """
        return self._sigma_c

    def get_theta(self):
        """
        Returns the parameter used to control how fast the precision
        distribution increases.
        """
        return self._sigma_c

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
        The hyper-parameter vector is ``[sigma_c, theta]``.
        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_sigma_c(x[0])
        self.set_theta(x[1])


