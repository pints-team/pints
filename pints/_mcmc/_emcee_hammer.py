#
# Emcee hammer MCMC
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
    Uses the differential evolution algorithm described in 
    Algorithm 2 in [1].
    
    For k in 1:N:
    - Draw a walker X_j at random from the complementary
        ensemble (i.e. the group of chains not including k)
        without replacement.
        
    - Sample z ~ g(z), (see below).

    - Set ``Y = X_j(t) + z[X_k(t) - X_j(t)]``.

    - Set ``q = z^{N-1} p(Y) / p(X_k(t))``.

    - Sample r ~ U(0,1).

    - if r <= q: X_k(t+1) = Y,
      else: X_k(t+1) = X_k(t).

    where g(z) is proportional to 1 / sqrt(z) if z is in [1 / a, a],
    or 0, otherwise, where a is a parameter set a priori (default is 2);
    N = number of chains (walkers).

    [1] "emcee: The MCMC Hammer", Daniel Foreman-Mackey, David W. Hogg,
    Dustin Lang, Jonathan Goodman, 2013, arXiv, 
    https://arxiv.org/pdf/1202.3665.pdf
    """

    def __init__(self, chains, x0, sigma0=None):
        super(DifferentialEvolutionMCMC, self).__init__(chains, x0, sigma0)

        # Need at least 3 chains
        if self._chains < 10:
            raise ValueError('Need at least 3 chains.')

        # Set initial state
        self._running = False

        # Current points and proposed points
        self._current = None
        self._current_logpdf = None
        self._proposed = None
        
        # Hyper parameter
        self._a = 2

    def ask(self):
        """ See :meth:`pints.MultiChainMCMC.ask()`. """
        # Initialise on first call
        if not self._running:
            self._initialise()
        
        # Propose new points
        if len(self._remaining) == 0:
            self._remaining = np.arange(self._chains)
        self._k = self._remaining[0]
        self._remaining = np.delete(self._remaining, 0)
        
        # pick j from the complementary ensemble
        j = np.random.randint(self._chains)
        while j == self._k:
            j = np.random.randint(self._chains)
        X_j = self._current[j]
        X_k = self._current[self._k]
        
        # sample Z from g[z] = (1/sqrt(Z)), if Z in [1/a, a],
        # 0 otherwise
        r = np.random.rand()
        self._Z = ((1 + (self._a - 1) * r)**2) / self._a
        self._proposed = X_j + self._Z * (X_k - X_j)

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
        
        # a range
        self._remaining = np.arange(self._chains)

        # Update sampler state
        self._running = True

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Emcee Hammer MCMC'

    def tell(self, fx):
        """ See :meth:`pints.MultiChainMCMC.tell()`. """
        # Check if we had a proposal
        if self._proposed is None:
            raise RuntimeError('Tell called before proposal was set.')

        # Ensure proposed_log_pdfs are numpy array
        proposed_log_pdfs = np.array(fx

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

        r_log = np.log(np.random.rand())
        q = ((self._chains - 1) * np.log(self._Z)  + proposed_log_pdfs[self._k] -
                self._current_log_pdfs[self._k])

        if q >= r_log:
            self._current[self._k] = self._proposed

        # Clear proposal
        self._proposed = None

        # Return samples to add to chains
        self._current.setflags(write=False)
        return self._current

    def set_a(self, a):
        """
        Sets the coefficient ``a`` used in updating the position of each
        chain.
        """
        a = float(a)
        if a <= 0:
            raise ValueError('The a parameter must be positive.')
        self._a = a

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 1

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[a]``.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_a(x[0])
