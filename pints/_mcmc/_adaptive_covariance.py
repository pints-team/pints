#
# Base class for adaptive covariance MCMC methods
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import numpy as np


class AdaptiveCovarianceMC(pints.SingleChainMCMC):
    """
    Base class for single chain MCMC methods that globally adapt a proposal
    covariance matrix when running, in order to control the acceptance rate.

    Each subclass should provide a method :meth:`_generate_proposal()` that
    will be called by :meth:`ask()`.

    Adaptation is implemented with three methods, which are called in
    sequence, at the end of every ``tell()``: :meth:`_adapt_mu()`,
    :meth:`_adapt_sigma()`, and :meth:`_adapt_internal()`.
    A basic implementation is provided for each, which extending methods can
    choose to override.

    Extends :class:`SingleChainMCMC`.
    """

    def __init__(self, x0, sigma0=None):
        super(AdaptiveCovarianceMC, self).__init__(x0, sigma0)

        # Current running status, used to initialise on first run and check
        # that certain methods are only called before or during run.
        self._running = False

        # Adaptive mode: disabled during initial phase
        self._adaptive = False

        # Current point and its log PDF
        self._current = None
        self._current_log_pdf = None

        # Proposed point
        self._proposed = None

        # Acceptance rate monitoring
        self._iterations = 0
        self._adaptations = 1

        # Target acceptance rate
        self._target_acceptance = None
        self.set_target_acceptance_rate()

        # Measured acceptance rate
        self._acceptance_count = 0
        self._acceptance_rate = 0

        # Parameters used in setting the proposal distributions
        # See update_mu() and update_sigma()
        self._mu = np.array(self._x0, copy=True)
        self._sigma = np.array(self._sigma0, copy=True)

        # Determines decay rate in adaptation
        self._eta = 0.6

        # Initial decay rate in adaptation
        self._gamma = 1

    def acceptance_rate(self):
        """
        Returns the current (measured) acceptance rate.
        """
        return self._acceptance_rate

    def _adapt_internal(self, accepted, log_ratio):
        """
        Called at the end of every ``tell()`` to adapt any internal parameters.

        Parameters
        ----------
        accepted : boolean
            Whether or not the proposal was accepted
        log_ratio : float
            The log of the ratio proposed log pdf / current log pdf
        """
        pass

    def _adapt_mu(self):
        """
        Called at the end of every ``tell()`` to adapt the current running mean
        used to calculate the sample covariance matrix of proposals.
        """
        self._mu = (1 - self._gamma) * self._mu + self._gamma * self._current

    def _adapt_sigma(self, log_ratio):
        """
        Called at the end of every ``tell()`` to adapt the covariance matrix
        used to generate proposals.

        Parameters
        ----------
        log_ratio
            The log of the ratio proposed log pdf / current log pdf.
        """
        dsigm = np.reshape(self._current - self._mu, (self._n_parameters, 1))
        self._sigma = ((1 - self._gamma) * self._sigma +
                       self._gamma * np.dot(dsigm, dsigm.T))

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """

        # Initialise on first call
        if not self._running:
            self._running = True

            # Store x0 as proposal, and set as read-only, so it can be passed
            # to user if it gets accepted.
            self._proposed = self._x0
            self._proposed.setflags(write=False)

        # Propose new point
        if self._proposed is None:
            # Let subclass generate proposal
            self._proposed = self._generate_proposal()

            # Set proposed as read-only, so it can be passed to user if it gets
            # accepted.
            self._proposed.setflags(write=False)

        # Return proposed point
        return self._proposed

    def eta(self):
        """
        Returns ``eta`` which controls the rate of adaptation decay
        ``adaptations**(-eta)``, where ``eta > 0`` to ensure asymptotic
        ergodicity.
        """
        return self._eta

    def _generate_proposal(self):
        """
        Should generate and return a proposed point.
        """
        raise NotImplementedError

    def in_initial_phase(self):
        """ See :meth:`pints.MCMCSampler.in_initial_phase()`. """
        return not self._adaptive

    def _log_init(self, logger):
        """ See :meth:`Loggable._log_init()`. """
        logger.add_float('Accept.')

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        logger.log(self._acceptance_rate)

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 1

    def needs_initial_phase(self):
        """ See :meth:`pints.MCMCSampler.needs_initial_phase()`. """
        return True

    def replace(self, current, current_log_pdf, proposed=None):
        """ See :meth:`pints.SingleChainMCMC.replace()`. """

        # At least one round of ask-and-tell must have been run
        if (not self._running) or self._current_log_pdf is None:
            raise RuntimeError(
                'Replace can only be used when already running.')

        # Check position
        current = pints.vector(current)
        if len(current) != self._n_parameters:
            raise ValueError('Point `current` has the wrong dimensions.')
        current.setflags(write=False)

        # Check log pdf
        current_log_pdf = float(current_log_pdf)

        # Check proposal
        if proposed is not None:
            proposed = pints.vector(proposed)
            if len(proposed) != self._n_parameters:
                raise ValueError('Point `proposed` has the wrong dimensions.')
            proposed.setflags(write=False)

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

            # Return first point for chain
            return self._current, self._current_log_pdf, True

        # Calculate log of the ratio of proposed and current log pdf
        # Can be used in adaptation, regardless of acceptance
        log_ratio = fx - self._current_log_pdf

        # Accept or reject the point
        accepted = False
        if np.isfinite(fx):
            u = np.log(np.random.uniform(0, 1))
            if u < log_ratio:
                accepted = True
                self._acceptance_count += 1

                # Update current point
                self._current = self._proposed
                self._current_log_pdf = fx

        # Calculate acceptance rate
        self._acceptance_rate = self._acceptance_count / self._iterations

        # Clear proposal
        self._proposed = None

        # Adapt covariance matrix
        if self._adaptive:

            # Set gamma based on number of adaptive iterations
            self._gamma = (self._adaptations + 1) ** -self._eta

            # Update the number of adaptations
            self._adaptations += 1

            # Update the proposal distribution
            self._adapt_mu()
            self._adapt_sigma(log_ratio)

            # Adapt
            self._adapt_internal(accepted, log_ratio)

        # Return current sample
        return self._current, self._current_log_pdf, accepted

