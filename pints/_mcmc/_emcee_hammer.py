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


class EmceeHammerMCMC(pints.MultiChainMCMC):
    """
    Uses the differential evolution algorithm "emcee: the MCMC hammer",
    described in Algorithm 2 in [1].

    For ``k`` in ``1:N``:
    - Draw a walker ``X_j`` at random from the "complementary ensemble" (the
      group of chains not including ``k``) without replacement.

    - Sample ``z ~ g(z)``, (see below).

    - Set ``Y = X_j(t) + z[X_k(t) - X_j(t)]``.

    - Set ``q = z^{N - 1} p(Y) / p(X_k(t))``.

    - Sample ``r ~ U(0, 1)``.

    - If ``r <= q``, set ``X_k(t + 1)`` equal to ``Y``, if not use ``X_k(t)``.

    Here, ``N`` is the number of chains (or walkers), and ``g(z)`` is
    proportional to ``1 / sqrt(z)`` if ``z`` is in  ``[1 / a, a]`` or to 0,
    otherwise (where ``a`` is a parameter with default value ``2``).

    [1] "emcee: The MCMC Hammer", Daniel Foreman-Mackey, David W. Hogg,
    Dustin Lang, Jonathan Goodman, 2013, arXiv,
    https://arxiv.org/pdf/1202.3665.pdf
    """

    def __init__(self, chains, x0, sigma0=None):
        super(EmceeHammerMCMC, self).__init__(chains, x0, sigma0)

        # Need at least 3 chains
        if self._chains < 3:
            raise ValueError('Need at least 3 chains.')

        # Set initial state
        self._running = False

        # Current points and proposed points
        self._current = None
        self._current_logpdf = None
        self._proposed = None

        # See docstring above
        self._a = 2

    def a(self):
        """
        Returns the coefficient ``a`` used in updating the position of each
        chain.
        """
        return self._a

    def ask(self):
        """ See :meth:`pints.MultiChainMCMC.ask()`. """
        # Initialise on first call
        if not self._running:
            self._initialise()

        # Propose new points
        if self._proposed is None:

            # Cycle through chains
            #TODO Do this without lists TODO
            if len(self._remaining) == 0:
                self._remaining = np.arange(self._chains)
            self._k = self._remaining[0]
            self._remaining = np.delete(self._remaining, 0)

            # Pick j from the complementary ensemble
            j = np.random.randint(self._chains - 1)
            if j >= self._k:
                j += 1
            x_j = self._current[j]
            x_k = self._current[self._k]

            # sample Z from g[z] = (1/sqrt(Z)), if Z in [1/a, a], 0 otherwise
            r = np.random.rand()
            self._z = ((1 + r * (self._a - 1))**2) / self._a
            self._proposed = x_j + self._z * (x_k - x_j)

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
        proposed_log_pdfs = np.array(fx)

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

        # Update only one of the chains #TODO WHY NOT ALL? WHAT's HAPPENING HERE?
        r_log = np.log(np.random.rand())
        q = ((self._chains - 1) * np.log(self._z)
            + proposed_log_pdfs[self._k]
            - self._current_log_pdfs[self._k])

        #TODO REWRITE AS MATRIX OPERATION? SEE DIFF EV MCMCMC
        if q >= r_log:
            next[self._k] = self._proposed
            next_log_pdfs[self._k] = self._proposed_log_pdfs[self._k]
            self._current = next
            self._current_log_pdfs = next_log_pdfs

        # Clear proposal
        self._proposed = None

        # Return samples to add to chains
        self._current.setflags(write=False)
        return self._current

    def set_a(self, a):
        """
        Sets the coefficient ``a`` used in updating the position of each chain.
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
