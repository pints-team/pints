#
# Population MCMC
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
# Some code in this file was adapted from Myokit (see http://myokit.org)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class PopulationMCMC(pints.SingleChainMCMC):
    """
    Creates a chain of samples from a target distribution, using the population
    MCMC (simulated tempering) routine described in algorithm 1 in [1]_.

    This method uses several chains internally, but only a single one is
    updated per iteration, and only a single one is returned at the end, hence
    this method is classified here as a single chain MCMC method.

    The algorithm goes through the following steps (after initialising ``N``
    internal chains):

    1. Mutation: randomly select chain ``i`` and update the chain using a
    Markov kernel that admits ``p_i`` as its invariant distribution.

    2. Exchange: Select another chain ``j`` at random from the remaining and
    swap the parameter vector of ``i`` and ``j`` with probability
    ``min(1, A)``,

    ``A = p_i(x_j) * p_j(x_i) / (p_i(x_i) * p_j(x_j))``

    where ``x_i`` and ``x_j`` are the current values of chains ``i`` and ``j``,
    respectively, where ``p_i = p(theta|data) ^ (1 - T_i)``, where
    ``p(theta|data)`` is the target distribution and ``T_i`` is bounded between
    ``[0, 1]`` and represents a tempering parameter.

    We use a range of ``T = (0,delta_T,...,1)``, where
    ``delta_T = 1 / num_temperatures``, and the chain with ``T_i = 0`` is the
    one whose target distribution we want to sample.

    Extends :class:`SingleChainMCMC`.

    References
    ----------
    .. [1] "On population-based simulation for static inference", Ajay Jasra,
           David A. Stephens and Christopher C. Holmes,
           Statistical Computing, 2007.
           https://doi.org/10.1007/s11222-007-9028-9
    """
    def __init__(self, x0, sigma0=None):
        super(PopulationMCMC, self).__init__(x0, sigma0)

        # Set initial state
        self._running = False

        # Current points, and the log pdfs of those points (_not_ the tempered
        # versions!)
        self._current = None
        self._current_log_pdfs = None

        # Single proposed point
        self._proposed = None

        # Inner chains
        self._chains = None

        #
        # Default settings
        #
        self._method = pints.HaarioBardenetACMC
        self._needs_initial_phase = True
        self._in_initial_phase = True

        # Temperature schedule
        self._schedule = None
        self.set_temperature_schedule()

        #
        # Logging
        #
        self._have_exchanged = False

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """
        # Initialise on first call
        if not self._running:
            self._initialise()

        # Propose new points
        if self._proposed is None:

            # Select chains to update and exchange/crossover
            n = len(self._chains)
            self._i, self._j = np.random.choice(n, 2, replace=False)

            # Propose new point
            self._proposed = self._chains[self._i].ask()

        # Return proposed point
        return self._proposed

    def current_log_pdf(self):
        """ See :meth:`SingleChainMCMC.current_log_pdf()`. """
        return self._current_log_pdfs[0]

    def in_initial_phase(self):
        """
        See :meth:`MCMCController.in_initial_phase()`.
        """
        return self._in_initial_phase

    def _initialise(self):
        """
        Initialises the routine before the first iteration.
        """
        if self._running:
            raise RuntimeError('Already initialised.')

        # Create inner chains
        n = len(self._schedule)
        self._chains = [self._method(self._x0, self._sigma0) for i in range(n)]

        # Set initial phase for methods that need it
        if self._needs_initial_phase:
            for chain in self._chains:
                chain.set_initial_phase(self._in_initial_phase)

        # Propose initial point
        self._current = None
        self._current_log_pdfs = None
        self._proposed = self._x0

        # Next chain to update
        self._i = 0

        # Next chain to exchange / crossover with
        self._j = 1

        # Ask all inner chains for first point (should be x0, so ignore!)
        for chain in self._chains:
            chain.ask()

        # Update sampler state
        self._running = True

    def _log_init(self, logger):
        """ See :meth:`Loggable._log_init()`. """
        n = len(self._schedule)
        logger.add_counter('i', max_value=n)
        logger.add_counter('j', max_value=n)
        logger.add_string('Ex.', width=3)

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        logger.log(self._i)
        logger.log(self._j)
        logger.log('yes' if self._have_exchanged else 'no')

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Population MCMC'

    def needs_initial_phase(self):
        """ See :meth:`pints.MCMCSampler.needs_initial_phase()`. """
        return self._needs_initial_phase

    def set_initial_phase(self, phase):
        """
        See :meth:`MCMCController.set_initial_phase()`.
        """
        self._in_initial_phase = bool(phase)
        if self._running:
            for chain in self._chains:
                chain.set_initial_phase(self._in_initial_phase)

    def set_temperature_schedule(self, schedule=10):
        """
        Sets a temperature schedule.

        If ``schedule`` is an ``int`` it is interpreted as the number of
        temperatures and a schedule is generated accordingly.

        If ``schedule`` is a list (or array) it is interpreted as a custom
        temperature schedule.
        """
        if self._running:
            raise RuntimeError(
                'Temperature schedule cannot be changed during run.')

        # Check type of schedule argument
        if np.isscalar(schedule):

            # Set using int
            schedule = int(schedule)
            if schedule < 2:
                raise ValueError(
                    'A schedule must contain at least two temperatures.')
            self._schedule = np.linspace(0, 0.95, schedule)
            self._schedule.setflags(write=False)

        else:

            # Set to custom schedule
            schedule = pints.vector(schedule)
            if len(schedule) < 2:
                raise ValueError(
                    'A schedule must contain at least two temperatures.')
            if schedule[0] != 0:
                raise ValueError(
                    'First element of temperature schedule must be 0.')

            # Check vector elements all between 0 and 1 (inclusive)
            if np.any(schedule < 0):
                raise ValueError('Temperatures must be non-negative.')
            if np.any(schedule > 1):
                raise ValueError('Temperatures cannot exceed 1.')

            # Store
            self._schedule = schedule

    def tell(self, fx):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """
        # Check if we had a proposal
        if self._proposed is None:
            raise RuntimeError('Tell called before proposal was set.')

        # Ensure fx is a float
        fx = float(fx)

        # First point?
        if self._current is None:
            # Pass to inner chains (ignore returned x0)
            for chain in self._chains:
                chain.tell(fx)

            # Always accept
            n = len(self._chains)
            self._current = np.array([self._x0] * n)

            # Store untempered logpdfs
            self._current_log_pdfs = np.array([fx] * n)

            # Clear proposal
            self._proposed = None

            # Return first point for chain 0
            sample = np.array(self._current[0], copy=False)
            sample.setflags(write=False)
            return sample

        # Perform mutation step (update one chain)

        # Update chain, get new sample
        sample = self._chains[self._i].tell(fx * (1 - self._schedule[self._i]))

        # Update current sample and untempered log pdf
        if np.any(sample != self._current[self._i]):
            self._current[self._i] = sample
            self._current_log_pdfs[self._i] = fx

        # Clear proposal
        self._proposed = None

        # Perform exchange step

        # Calculate current tempered likelihoods
        pii = (1 - self._schedule[self._i]) * self._current_log_pdfs[self._i]
        pjj = (1 - self._schedule[self._j]) * self._current_log_pdfs[self._j]

        # Calculate proposed log likelihoods
        pij = (1 - self._schedule[self._i]) * self._current_log_pdfs[self._j]
        pji = (1 - self._schedule[self._j]) * self._current_log_pdfs[self._i]

        # Accept/reject exchange step
        self._have_exchanged = False
        if np.isfinite(pij) and np.isfinite(pji):
            u = np.log(np.random.uniform(0, 1))
            if u < pij + pji - (pii + pjj):
                self._chains[self._i].replace(self._current[self._j], pij)
                self._chains[self._j].replace(self._current[self._j], pji)
                self._have_exchanged = True

        # Return new point for chain 0
        sample = np.array(self._current[0], copy=False)
        sample.setflags(write=False)
        return sample

    def temperature_schedule(self):
        """
        Returns the temperature schedule used in the tempering algorithm. Each
        temperature ``T`` pertains to particular chain whose stationary
        distribution is ``p(theta|data) ^ (1 - T)``.
        """
        return self._schedule
