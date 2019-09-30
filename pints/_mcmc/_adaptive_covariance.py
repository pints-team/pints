#
# Base class for adaptive covariance MCMC methods
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


class AdaptiveCovarianceMC(pints.SingleChainMCMC):
    """
    Base class for single chain MCMC methods that adapt a covariance matrix
    when running, in order to control the acceptance rate.

    In all cases ``eta`` is used to control decay of adaptation.
    """
    def __init__(self, x0, sigma0=None):
        super(AdaptiveCovarianceMC, self).__init__(x0, sigma0)

        # Set initial state
        self._running = False

        # Set initial mu and sigma
        self._mu = np.array(self._x0, copy=True)
        self._sigma = np.array(self._sigma0, copy=True)

        # initial number of adaptations (must start at 1 otherwise fails)
        self._adaptations = 1
        # initial decay rate in adaptation
        self._gamma = 1
        # determines decay rate in adaptation
        self._eta = 0.6

        # Acceptance rate monitoring
        self._iterations = 0

        # Default settings
        self.set_target_acceptance_rate()

        # Adaptive mode: disabled during initial phase
        self._adaptive = False

        self._current = None
        self._current_log_pdf = None
        self._proposed = None
        self._log_acceptance_ratio = None

    def acceptance_rate(self):
        """
        Returns the current (measured) acceptance rate.
        """
        return self._acceptance

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """
        # Initialise on first call
        if not self._running:
            self._running = True
            self._proposed = self._x0

    def current_log_pdf(self):
        """ See :meth:`SingleChainMCMC.current_log_pdf()`. """
        return self._current_log_pdf

    def eta(self):
        """
        Returns ``eta`` which controls the rate of adaptation decay
        ``adaptations**(-eta)``, where ``eta > 0`` to ensure asymptotic
        ergodicity.
        """
        return self._eta

    def in_initial_phase(self):
        """ See :meth:`pints.MCMCSampler.in_initial_phase()`. """
        return not self._adaptive

    def _log_init(self, logger):
        """ See :meth:`Loggable._log_init()`. """
        logger.add_float('Accept.')

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        logger.log(self._acceptance)

    def needs_initial_phase(self):
        """ See :meth:`pints.MCMCSampler.needs_initial_phase()`. """
        return True

    def replace(self, current, current_log_pdf, proposed=None):
        """ See :meth:`pints.SingleChainMCMC.replace()`. """

        # At least one round of ask-and-tell must have been run
        if (not self._running) or self._current_log_pdf is None:
            raise RuntimeError(
                'Replace can only be used when already running.')

        # Check values
        current = pints.vector(current)
        if len(current) != self._n_parameters:
            raise ValueError('Point `current` has the wrong dimensions.')
        current_log_pdf = float(current_log_pdf)
        if proposed is not None:
            proposed = pints.vector(proposed)
            if len(proposed) != self._n_parameters:
                raise ValueError('Point `proposed` has the wrong dimensions.')

        # Store
        self._current = current
        self._current_log_pdf = current_log_pdf
        self._proposed = proposed

    def set_eta(self, eta):
        """
        Updates ``eta`` which controls the rate of adaptation decay
        ``adaptations**(-eta)``, where ``eta > 0`` to ensure asymptotic
        ergodicity.
        """
        if eta <= 0:
            raise ValueError('eta should be greater than zero')
        self._eta = eta

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[eta]``.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_eta(x[0])

    def set_initial_phase(self, initial_phase):
        """ See :meth:`pints.MCMCSampler.set_initial_phase()`. """
        # No adaptation during initial phase
        self._adaptive = not bool(initial_phase)

    def set_target_acceptance_rate(self, rate=0.234):
        """
        Sets the target acceptance rate.
        """
        rate = float(rate)
        if rate <= 0:
            raise ValueError('Target acceptance rate must be greater than 0.')
        elif rate > 1:
            raise ValueError('Target acceptance rate cannot exceed 1.')
        self._target_acceptance = rate

    def target_acceptance_rate(self):
        """
        Returns the target acceptance rate.
        """
        return self._target_acceptance

    def tell(self, fx):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """

        # Check if we had a proposal
        if self._proposed is None:
            raise RuntimeError('Tell called before proposal was set.')

        # Ensure fx is a float
        fx = float(fx)

        # Increase iteration count
        self._iterations += 1

        # First point?
        if self._current is None:
            if not np.isfinite(fx):
                raise ValueError(
                    'Initial point for MCMC must have finite logpdf.')

            # Accept
            self._current = self._proposed
            self._current_log_pdf = fx

            # Clear proposal
            self._proposed = None

            # Set r to zero
            self._log_acceptance_ratio = float('-Inf')

            # Return first point for chain
            return self._current

        # Check if the proposed point can be accepted
        self._log_acceptance_ratio = fx - self._current_log_pdf

    def _update_mu(self):
        """
        Updates the current running mean used to calculate the sample
        covariance matrix of proposals.
        """
        raise NotImplementedError

    def _update_sigma(self):
        """
        Updates the covariance matrix used to generate proposals.
        """
        raise NotImplementedError
