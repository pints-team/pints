#
# emcee Hammer MCMC
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
    *Extends:* :class:`MultiChainMCMC`

    Uses the differential evolution algorithm described in 
    Algorithm 3 in [1].

    Initially, the group of chains are split into two random
    subgroups, of equal sizes. Then we update chain k
    in each group according to:

    - Draw a walker X_j at random from the complementary
        ensemble (i.e. the other group) without replacement.
        
    - Sample z ~ g(z), (see below).

    - Set ``Y = X_j(t) + z[X_k(t) - X_j(t)]``.

    - Set ``q = z^{N-1} p(Y) / p(X_k(t))``.

    - Sample r ~ U(0,1).

    - if r <= q: X_k(t+1) = Y,
      else: X_k(t+1) = X_k(t).

    where g(z) is proportional to 1 / sqrt(z) if z is in [1/a, a],
    or 0, otherwise, where a is a parameter set a priori (default is 2);
    N = number of chains (walkers).

    [1] "emcee: The MCMC Hammer", Daniel Foreman-Mackey, David W. Hogg,
    Dustin Lang, Jonathan Goodman, 2013, arXiv, 
    https://arxiv.org/pdf/1202.3665.pdf
    """
    def __init__(self, chains, x0, sigma0=None):
        super(EmceeHammerMCMC, self).__init__(chains, x0, sigma0)

        if chains < 10:
            raise ValueError('Need at least 10 chains.')
        if chains % 2 != 0:
            raise ValueError('The number of chains must be an even integer.')

        # Set initial state
        self._running = False

        # Current points and proposed points
        self._current = None
        self._current_logpdf = None
        self._proposed = None
        self._n_chains = chains
        self._n_chains_half = int(chains / 2)

        # Default
        self._a = 2
            
    def set_n_chains(self, n):
        """
        Sets the number of chains (walkers) to run,
        and splits these into two equal sized groups.
        """
        if n < 10:
            raise ValueError('Must run with at least 10 chains.')
        if n % 2 != 0:
            raise ValueError('The number of chains must be an even integer.')
        self._n_chains = n
        self._n_chains_half = int(n / 2)

    def _split_chains(self):
        """
        Splits chains into two roughly equal sized groups.
        """
        self._group_0 = np.arange(0, int(float(self._n_chains) / 2.0))
        self._group_1 = np.arange(int(float(self._n_chains) / 2.0), self._n_chains)

    def ask(self):
        """ See :meth:`pints.MultiChainMCMC.ask()`.
            Updates the positin of all of the chains (walkers)
            within the group being considered as per Algorithm 3 in [1].
            
            [1] "emcee: The MCMC Hammer", Daniel Foreman-Mackey, David W. Hogg,
            Dustin Lang, Jonathan Goodman, 2013, arXiv,
            https://arxiv.org/pdf/1202.3665.pdf
        """
        # Initialise on first call
        if not self._running:
            self._initialise()
        
        # Only runs after first tell
        if self._proposed is None:
            # If all updates for given group are done
            # move onto the other
            self._proposed = np.zeros(self._current.shape)
            self._k = np.zeros(self._n_chains_half, dtype='int')
            self._z = np.zeros(self._n_chains_half)
            
            if len(self._remaining) == 0:
                # Select other
                self._current_group = (self._current_group + 1) % 2
                if self._current_group == 0:
                    self._remaining = self._group_1
                    self._other = self._group_0
                else:
                    self._remaining = self._group_0
                    self._other = self._group_1

            # Make proposals for all walkers within a given group,
            # and save z for each of those
            for i in range(0, self._n_chains_half):
                # Randomly select a chain from the other group
                self._remaining = np.random.permutation(self._remaining)
                j = self._remaining[len(self._remaining) - 1]
                self._remaining = np.delete(self._remaining, len(self._remaining) - 1)
                self._k[i] = self._other[0]
                self._other = np.delete(self._other, 0)
                X_j = self._current[j]
                X_k = self._current[self._k[i]]

                # Sample from g(z) = 1/sqrt(z) if z in [1/a,a], 0 otherwise
                # using inverse transform sampling:
                # inv-CDF(r,a) = (1 + (a - 1)r)^2/a
                r = np.random.rand()
                self._z[i] = ((1 + (self._a - 1) * r)**2) / self._a

                # Create proposal
                print(X_k)
                self._proposed[self._k[i]] = X_j + self._z[i] * (X_k - X_j)

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
        
        # Split chains into two groups
        self._split_chains()

        # Set current group being updated
        self._current_group = 0

        # Set current remaining chains as group 1 (i.e.
        # the second) as a starting complementary ensemble
        self._remaining = self._group_1
        self._other = self._group_0
        
        # Set initial z
        self._z = None

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'emcee Hammer MCMC'

    def tell(self, proposed_log_pdfs):
        """ See :meth:`pints.MultiChainMCMC.tell()`. """
        # Check if we had a proposal
        if self._proposed is None:
            raise RuntimeError('Tell called before proposal was set.')

        # Ensure proposed_log_pdfs are numpy array
        print(proposed_log_pdfs)
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
        
        current_temp = np.zeros(self._current.shape)
        # For the chains being updated
        for i in range(0, self._n_chains_half):
            q = (self._z[i]**(self._n_chains - 1))  * proposed_log_pdfs[self._k[i]] /\
                self._current_log_pdfs[self._k[i]]
            r = np.random.rand()

            if r <= q:
                current_temp[self._k[i]] = proposed_log_pdfs[self._k[i]]

        # For the other half
        for i in range(0, self._n_chains_half):
            current_temp[(self._k[i] + 5) % 10] = self._current[(self._k[i] + 5) % 10]

        self._current = current_temp
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
