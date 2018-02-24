#
# Differential evolution MCMC
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


class DifferentialEvolutionMCMC(pints.MultiChainMCMC):
    """
    *Extends:* :class:`MultiChainMCMC`

    Uses differential evolution MCMC as described in [1] to do posterior
    sampling from the posterior.

    In each step of the algorithm ``n`` chains are evolved using the evolution
    equation::

        x_proposed = x[i,r] + gamma * (X[i,r1] - x[i,r2]) + epsilon

    where ``r1`` and ``r2`` are random chain indices chosen (without
    replacement) from the ``n`` available chains, which must not equal ``i`` or
    each other, where ``i`` indicates the current time  step, and
    ``epsilon ~ N(0,b)`` where ``d`` is the dimensionality of the parameter
    vector.

    If ``x_proposed / x[i,r] > u ~ U(0,1)``, then
    ``x[i+1,r] = x_proposed``; otherwise, ``x[i+1,r] = x[i]``.

    [1] "A Markov Chain Monte Carlo version of the genetic algorithm
    Differential Evolution: easy Bayesian computing for real parameter spaces"
    Cajo J. F. Ter Braak (2006) Statistical Computing
    """
    def __init__(self, chains, x0, sigma0=None):
        super(DifferentialEvolutionMCMC, self).__init__(chains, x0, sigma0)

        # Need at least 3 chains
        if self._chains < 3:
            raise ValueError('Need at least 3 chains.')

        # Set initial state
        self._running = False

        # Current points and proposed points
        self._current = None
        self._current_logpdf = None
        self._proposed = None

        #
        # Default settings
        #

        # Gamma
        self._gamma = 2.38 / np.sqrt(2 * self._dimension)

        # Normal proposal std.
        self._b = 0.01

    def ask(self):
        """ See :meth:`pints.MultiChainMCMC.ask()`. """
        # Initialise on first call
        if not self._running:
            self._initialise()

        # Propose new points
        if self._proposed is None:
            self._proposed = np.zeros(self._current.shape)
            for j in range(self._chains):
                r1, r2 = r_draw(j, self._chains)
                self._proposed[j] = (
                    self._current[j]
                    + self._gamma * (self._current[r1] - self._current[r2])
                    + np.random.normal(0, self._b * self._mu, self._mu.shape)
                )

            # Set as read only
            self._proposed.setflags(write=False)

        # Return proposed points
        return self._proposed

    def _initialise(self):
        """
        Initialises the routine before the first iteration.
        """
        if self._running:
            raise RuntimeError('Already initialised.')

        # Propose x0 as first points
        self._current = None
        self._current_log_pdfs = None
        self._proposed = self._x0

        # Set mu
        #TODO: Should this be a user setting?
        self._mu = np.mean(self._x0, axis=0)

        # Update sampler state
        self._running = True

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Differential Evolution MCMC'

    def tell(self, proposed_log_pdfs):
        """ See :meth:`pints.MultiChainMCMC.tell()`. """
        # Check if we had a proposal
        if self._proposed is None:
            raise RuntimeError('Tell called before proposal was set.')

        # Ensure proposed_log_pdfs are numpy array
        proposed_log_pdfs = np.array(proposed_log_pdfs)

        # First points?
        if self._current is None:
            if not np.all(np.isfinite(proposed_log_pdfs)):
                raise ValueError(
                    'Initial points for MCMC must have finite logpdf.')

            # Accept
            self._current = self._proposed
            self._current_log_pdfs = proposed_log_pdfs

            # Clear proposal
            self._proposed = None

            # Return first samples for chains
            return self._current

        # Perform iteration
        next = np.array(self._current, copy=True)
        next_log_pdfs = np.array(self._current_log_pdfs, copy=True)

        # Sample uniform numbers
        u = np.log(np.random.uniform(size=self._chains))

        # Get chains to be updated
        i = u < (proposed_log_pdfs - self._current_log_pdfs)

        # Update
        next[i] = self._proposed[i]
        next_log_pdfs[i] = proposed_log_pdfs[i]
        self._current = next
        self._current_log_pdfs = next_log_pdfs

        # Clear proposal
        self._proposed = None

        # Return samples to add to chains
        self._current.setflags(write=False)
        return self._current

    def set_normal_scale_coefficient(self, b):
        """
        Sets the normal scale coefficient ``b`` used in updating the position
        of each chain.
        """
        b = float(b)
        if b < 0:
            raise ValueError('Normal scale coefficient must be non-negative.')
        self._b = b

    def set_gamma(self, gamma):
        """
        Sets the coefficient ``gamma`` used in updating the position of each
        chain.
        """
        gamma = float(gamma)
        if gamma < 0:
            raise ValueError('Gamma must be non-negative.')
        self._gamma = gamma


def r_draw(i, num_chains):
    #TODO: Needs a docstring!
    r1, r2 = np.random.choice(num_chains, 2, replace=False)
    while(r1 == i or r2 == i or r1 == r2):
        r1, r2 = np.random.choice(num_chains, 2, replace=False)
    return r1, r2
