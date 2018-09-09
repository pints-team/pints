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

        # Need at least 3 chains
        if self._chains < 4:
            raise ValueError('Need at least 4 chains.')
        if self._chains % 2 != 0:
            raise ValueError('The number of chains must be an even integer.')
        self._chains_half = int(float(self._chains) / 2.0)
        
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
        
    def _split_chains(self):
        """
        Splits chains into two equal sized groups.
        """
        self._group_0 = np.arange(0, int(float(self._chains) / 2.0))
        self._group_1 = np.arange(int(float(self._chains) / 2.0), self._chains)

    def ask(self):
        """ See :meth:`pints.MultiChainMCMC.ask()`.
            Updates the positin of the chains (walkers)
            within the group being considered as per Algorithm 3 in [1]. The
            returned group of positions is theferefore of length
            equal to half the number of total chains
            
            [1] "emcee: The MCMC Hammer", Daniel Foreman-Mackey, David W. Hogg,
            Dustin Lang, Jonathan Goodman, 2013, arXiv,
            https://arxiv.org/pdf/1202.3665.pdf
        """
        # Initialise on first call
        if not self._running:
            self._initialise()
        
        if len(self._remaining) == self._chains_half and self._current_group == 0:
            self._count = 0
            self._proposed_all = np.zeros(self._current.shape)
            self._current_copy = np.copy(self._current)
        
        # Only runs after first tell
        if self._proposed is None:
            # If all updates for given group are done
            # move onto the other
            self._proposed = np.zeros(self._chains_half, self._dimension)
            self._k = np.zeros(self._chains_half, dtype='int')
            self._z = np.zeros(self._chains_half)
            
            if len(self._remaining) == 0:
                # Select other
                self._current_group = (self._current_group + 1) % 2
                if self._current_group == 0:
                    self._other_remaining = self._group_1
                    self._current_members= self._group_0
                else:
                    self._other_remaining = self._group_0
                    self._current_members = self._group_1

            # Make proposals for all walkers within a given group,
            # and save z for each of those
            for i in range(0, self._chains_half):
                # Randomly select a chain from the other group
                self._other_remaining = np.random.permutation(self._other_remaining)
                j = self._other_remaining[len(self._other_remaining) - 1]
                self._other_remaining = np.delete(self._other_remaining,
                                                  len(self._other_remaining) - 1)
                k = self._current_members[i]
                X_j = self._current_copy[j]
                X_k = self._current_copy[k]
                self._indices[self._count] = k

                # Sample from g(z) = 1/sqrt(z) if z in [1/a,a], 0 otherwise
                # using inverse transform sampling:
                # inv-CDF(r,a) = (1 + (a - 1)r)^2/a
                r = np.random.rand()
                self._z[i] = ((1 + (self._a - 1) * r)**2) / self._a

                # Create proposal
                self._proposed[i] = X_j + self._z[i] * (X_k - X_j)
                self._proposed_all[self._count] = self._proposed_all[i]
                self._count += 1

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
        self._other_remaining = self._group_1
        self._current_members = self._group_0
        self._count = 0
        self._indices = np.zeros(self._chains)
        
        # Set initial z
        self._z = None

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Differential Evolution MCMC'

    def tell(self, fx):
        """ See :meth:`pints.MultiChainMCMC.tell()`. 
        Accepts or rejects proposals for half the chains.
        If first half is updated then return None; otherwise
        return all the chains.
        """
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
            self._current_log_pdfs = fx

            # Clear proposal
            self._proposed = None

            # Return first samples for chains
            return self._current

        # For the chains being updated
        if self._current_group == 0:
            counter = 0
        else:
            counter = self._chains_half
        
        for i in range(0, self._chains_half):
            r = ((self._z[i]**(self._chains - 1))  * 
                 fx[i] \ self._current_log_pdfs[i])
            u = np.random.rand()

            if r <= q:
                self._current[self._indices[i]] = self._proposed

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

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 2

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[gamma, normal_scale_coefficient]``.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_gamma(x[0])
        self.set_normal_scale_coefficient(x[1])


def r_draw(i, num_chains):
    # TODO: Needs a docstring!
    r1, r2 = np.random.choice(num_chains, 2, replace=False)
    while(r1 == i or r2 == i or r1 == r2):
        r1, r2 = np.random.choice(num_chains, 2, replace=False)
    return r1, r2
