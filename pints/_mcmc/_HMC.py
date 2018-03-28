#
# HMC MCMC method
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


class HMCMCMC(pints.SingleChainAdaptiveMCMC):
    """
    *Extends:* :class:`SingleChainAdaptiveMCMC`

    Implements Hamiltonian Monte Caro as described in [1].

    Uses a physical analogy of a particle moving across a landscape under
    Hamiltonian dynamics to aid efficient exploration of parameter space.
    Introduces an auxilary variable -- the momentum (p_i) of a particle
    moving in dimension i of negative log posterior space -- which supplements
    the position (q_i) of the particle in parameter space. The particle's
    motion is dictated by solutions to Hamilton's equations,

    dq_i/dt =   partial_d H/partial_d p_i,
    dp_i/dt = - partial_d H/partial_d q_i.

    The Hamiltonian is given by,

    H(q,p) =       U(q)       +        KE(p)  
           = -log(p(q|X)p(q)) + Sigma_i=1^d p_i^2/2m_i,

    where d is the dimensionality of model and m_i is the 'mass' given
    to each particle (often chosen to be 1 as default).

    To numerically integrate Hamilton's equations, it is essential to use a
    sympletic discretisation routine, of which the most typical approach is
    the leapfrog method,

    p_i(t + epsilon) = p_i(t) + epsilon dp_i(t)/dt = p_i(t) - epsilon partial_d U(q)/d_q_i,
    q_i(t + epsilon) = q_i(t) + epsilon dq_i(t)/dt = q_i(t) + epsilon p_i(t) / m_i.

    [1] MCMC using Hamiltonian dynamics
    Radford M. Neal, Chapter 5 of the Handbook of Markov Chain Monte
    Carlo by Steve Brooks, Andrew Gelman, Galin Jones, and Xiao-Li Meng.

    """
    def __init__(self, x0, sigma0=None):
        super(SingleChainAdaptive, self).__init__(x0, sigma0)

        # Set initial state
        self._running = False

        # Current point and proposed point
        self._current = None
        self._current_log_pdf = None
        self._proposed = None

        # Default settings
        self.set_target_acceptance_rate()

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

        # Propose new point
        if self._proposed is None:

            # Note: Normal distribution is symmetric
            #  N(x|y, sigma) = N(y|x, sigma) so that we can drop the proposal
            #  distribution term from the acceptance criterion
            self._proposed = np.random.multivariate_normal(
                self._current, np.exp(self._loga) * self._sigma)

            # Set as read-only
            self._proposed.setflags(write=False)

        # Return proposed point
        return self._proposed

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

        # Acceptance rate monitoring
        self._iterations = 0
        self._acceptance = 0

        # Update sampler state
        self._running = True

    def _log_init(self, logger):
        """ See :meth:`Loggable._log_init()`. """
        logger.add_float('Accept.')

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        logger.log(self._acceptance)

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Adaptive covariance MCMC'

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

            # Return first point for chain
            return self._current

        # Check if the proposed point can be accepted
        accepted = 0
        if np.isfinite(fx):
            u = np.log(np.random.uniform(0, 1))
            if u < fx - self._current_log_pdf:
                accepted = 1
                self._current = self._proposed
                self._current_log_pdf = fx

        # Clear proposal
        self._proposed = None

        # Adapt covariance matrix
        if self._adaptation:
            # Set gamma based on number of adaptive iterations
            gamma = self._adaptations ** -0.6
            self._adaptations += 1

            # Update mu, log acceptance rate, and covariance matrix
            self._mu = (1 - gamma) * self._mu + gamma * self._current
            self._loga += gamma * (accepted - self._target_acceptance)
            dsigm = np.reshape(self._current - self._mu, (self._dimension, 1))
            self._sigma = (
                (1 - gamma) * self._sigma + gamma * np.dot(dsigm, dsigm.T))

        # Update acceptance rate (only used for output!)
        self._acceptance = ((self._iterations * self._acceptance + accepted) /
                            (self._iterations + 1))

        # Increase iteration count
        self._iterations += 1

        # Return new point for chain
        return self._current

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

    def set_target_acceptance_rate(self, rate=0.3):
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
        return self._target_acceptance_rate

