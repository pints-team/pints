#
# Adaptive covariance MCMC method
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
# Some code in this file was adapted from Myokit (see http://myokit.org)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class MetropolisHastingMCMC(pints.SingleChainMCMC):
    """
    *Extends:* :class:`SingleChainAdaptiveMCMC`

    Metropolis Hasting MCMC, as described in [1].

    # TODO Add description.
    Standard Metropolis Hasting using multivariate Normal distribution as
    proposal step, also known as Metropolis Random Walk MCMC.

    # TODO: Add citation.
    [1]
    """
    def __init__(self, x0, sigma0=None):
        super(MetropolisHastingMCMC, self).__init__(x0, sigma0)

        # Set initial state
        self._running = False
        self._ready_for_tell = False
        self._need_first_point = True

    def acceptance_rate(self):
        """
        Returns the current (measured) acceptance rate.
        """
        return self._acceptance

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """
        # Initialise on first call
        if not self._running:
            self._initialise()

        # After this method we're ready for tell()
        self._ready_for_tell = True

        # Need evaluation of first point?
        if self._need_first_point:
            return self._current

        # Propose new point
        # Note: Normal distribution is symmetric
        #  N(x|y, sigma) = N(y|x, sigma) so that we can drop the proposal
        #  distribution term from the acceptance criterion
        # TODO: Maybe allow general proposal disbution which has sampling
        #       method. This should be the "Metropolis-Hasting" rather than
        #       "Metropolis Random Walk".
        self._proposed = np.random.multivariate_normal(
            self._current, self._sigma0)

        # Return proposed point
        self._proposed.setflags(write=False)
        return self._proposed

    def _initialise(self):
        """
        Initialises the routine before the first iteration.
        """
        if self._running:
            raise Exception('Already initialised.')

        # Set current sample
        self._current = self._x0
        self._current_log_pdf = float('inf')

        # Iteration counts (for acceptance rate)
        self._iterations = 0

        # Initial acceptance rate
        self._acceptance = 0

        # Update sampler state
        self._running = True

    def _log_init(self, logger):
        """ See :meth:`Loggable._log_init`. """
        logger.add_float('Accept.')

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write`. """
        logger.log(self._acceptance)

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Metropolis hasting MCMC'

    def tell(self, fx):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """

        # First point?
        if self._need_first_point:
            if not np.isfinite(fx):
                raise ValueError(
                    'Initial point for MCMC must have finite logpdf.')
            self._current_log_pdf = fx

            # Increase iteration count
            self._iterations += 1

            # Update state
            self._need_first_point = False
            self._ready_for_tell = False

            # Return first point for chain
            return self._current

        # Check if the proposed point can be accepted
        accepted = 0
        if np.isfinite(fx):
            u = np.log(np.random.uniform(0, 1))
            # TODO: Maybe allow temperature annealing
            if u < fx - self._current_log_pdf:
                accepted = 1
                self._current = self._proposed
                self._current_log_pdf = fx

        # Update acceptance rate (only used for output!)
        self._acceptance = ((self._iterations * self._acceptance + accepted) /
                            (self._iterations + 1))

        # Increase iteration count
        self._iterations += 1

        # Update state
        self._ready_for_tell = False

        # Return new point for chain
        return self._current
