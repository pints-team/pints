#
# Adaptive covariance MCMC method
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class AdaptiveCovarianceMCMC(pints.SingleChainMCMC):
    """
    Adaptive covariance MCMC general class covering a range of
    methods where the covariance matrix is adapted in MCMC runs.
    In all cases self._adaptations ^ -eta is used to control decay
    of adaptation

    *Extends:* :class:`SingleChainMCMC`
    """
    def __init__(self, x0, sigma0=None):
        super(AdaptiveCovarianceMCMC, self).__init__(x0, sigma0)

        # Set initial state
        self._running = False

        # Current point and proposed point
        self._current = None
        self._current_log_pdf = None
        self._proposed = None

        # Default settings
        self.set_target_acceptance_rate()

        # Adaptive mode: disabled during initial phase
        self._adaptive = False

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

    def _initialise(self):
        """
        Initialises the routine before the first iteration.
        """
        if self._running:
            raise RuntimeError('Already initialised.')

        # Propose x0 as first point
        self._current = None
        self._current_log_pdf = None
        self._proposed = self._x0

        # Set initial mu and sigma
        self._mu = np.array(self._x0, copy=True)
        self._sigma = np.array(self._sigma0, copy=True)

        # Adaptation
        self._loga = 0
        self._adaptations = 2
        self._gamma = 1
        self._eta = 0.6

        # Acceptance rate monitoring
        self._iterations = 0
        self._acceptance = 0

        # Update sampler state
        self._running = True
        
        # Localised AM
        self._localised = False
        self._initial_fit = True
    
    def set_eta(self, eta):
        """
        Updates eta which controls the rate of
        adaptation decay by self._adaptations ^ -eta, where
        eta > 0 to ensure asymptotic ergodicity
        """
        if eta <= 0:
            raise ValueError('eta should be greater than zero')
        self._eta = eta

    def in_initial_phase(self):
        """ See :meth:`pints.MCMCSampler.in_initial_phase()`. """
        return not self._adaptive

    def _log_init(self, logger):
        """ See :meth:`Loggable._log_init()`. """
        logger.add_float('Accept.')

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        logger.log(self._acceptance)

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Adaptive covariance MCMC'

    def needs_initial_phase(self):
        """ See :meth:`pints.MCMCSampler.needs_initial_phase()`. """
        return True

    def set_initial_phase(self, initial_phase):
        """ See :meth:`pints.MCMCSampler.set_initial_phase()`. """
        # No adaptation during initial phase
        self._adaptive = not bool(initial_phase)

    def tell(self, fx):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """
        # Check if we had a proposal
        if self._proposed is None:
            raise RuntimeError('Tell called before proposal was set.')

        # Ensure fx is a float
        fx = float(fx)

        # First point?
        if self._current is None:
            if not np.isfinite(fx):
                raise ValueError(
                    'Initial point for MCMC must have finite logpdf.')

            # Accept
            self._current = self._proposed
            self._current_log_pdf = fx

            # Increase iteration count
            self._iterations += 1

            # Clear proposal
            self._proposed = None
            
            # Set alpha prob to zero
            self._alpha = 0

            # Return first point for chain
            return self._current

        # Check if the proposed point can be accepted
        self._accepted = 0
        r = fx - self._current_log_pdf
        self._alpha = np.minimum(1, np.exp(r))
        self._X = self._current
        self._Y = self._proposed
        if np.isfinite(fx):
            u = np.log(np.random.uniform(0, 1))
            if u < fx - self._current_log_pdf:
                self._accepted = 1
                self._current = self._proposed
                self._current_log_pdf = fx

        # Clear proposal
        self._proposed = None

        # Adapt covariance matrix
        if self._adaptive:
            # For localised AM
            if self._localised:
                self._fit_gaussian_mixture()
                self._initial_fit = False
            else:
                # Update mu and covariance matrix
                self._update_mu()
                self._update_sigma()

            # Set gamma based on number of adaptive iterations
            self._gamma = self._adaptations ** -self._eta
            self._adaptations += 1

            

        # Update acceptance rate (only used for output!)
        self._acceptance = ((self._iterations * self._acceptance + self._accepted) /
                            (self._iterations + 1))

        # Increase iteration count
        self._iterations += 1

    def _update_mu(self):
        """
        Updates the current running mean used to calculate the sample
        covariance matrix of proposals. Note that this default is overidden in
        some of the methods
        """
        self._mu = (1 - self._gamma) * self._mu + self._gamma * self._current
        
    def _update_sigma(self):
        """
        Updates the covariance matrix used to generate proposals.
        Note that this default is overidden in some of the methods
        """
        dsigm = np.reshape(self._current - self._mu, (self._dimension, 1))
        self._sigma = ((1 - self._gamma) * self._sigma + self._gamma * np.dot(dsigm, dsigm.T))

    def replace(self, x, fx):
        """ See :meth:`pints.SingleChainMCMC.replace()`. """
        # Must already be running
        if not self._running:
            raise RuntimeError(
                'Replace can only be used when already running.')

        # Must be after tell, before ask
        if self._proposed is not None:
            raise RuntimeError(
                'Replace can only be called after tell / before ask.')

        # Check values
        x = pints.vector(x)
        if not len(x) == len(self._current):
            raise ValueError('Dimension mismatch in `x`.')
        fx = float(fx)

        # Store
        self._current = x
        self._current_log_pdf = fx

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

    def _fit_gaussian_mixture(self):
        """
        Fits a Gaussian mixture distribution by updating
        componentwise the means, covariance matrices,
        weights and lamdas (eq. 36 and 37 in Andrieu &
        Thoms 2008)
        """
        raise NotImplementedError
        

