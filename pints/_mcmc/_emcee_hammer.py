#
# Emcee hammer MCMC
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import numpy as np


class EmceeHammerMCMC(pints.MultiChainMCMC):
    """
    Uses the differential evolution algorithm "emcee: the MCMC hammer",
    described in Algorithm 2 in [1]_.

    For ``k`` in ``1:N``:

    - Draw a walker ``X_j`` at random from the "complementary ensemble" (the
      group of chains not including ``k``) without replacement.

    - Sample ``z ~ g(z)``, (see below).

    - Set ``Y = X_j(t) + z[X_k(t) - X_j(t)]``.

    - Set ``q = z^{d - 1} p(Y) / p(X_k(t))``.

    - Sample ``r ~ U(0, 1)``.

    - If ``r <= q``, set ``X_k(t + 1)`` equal to ``Y``, if not use ``X_k(t)``.

    Here, ``N`` is the number of chains (or walkers), ``d`` is the
    dimensionality of the space, and ``g(z)`` is proportional to
    ``1 / sqrt(z)`` if ``z`` is in  ``[1 / a, a]`` or to 0, otherwise (where
    ``a`` is a parameter with default value ``2``).

    References
    ----------
    .. [1] "emcee: The MCMC Hammer", Daniel Foreman-Mackey, David W. Hogg,
           Dustin Lang, Jonathan Goodman, 2013, arXiv,
           https://arxiv.org/pdf/1202.3665.pdf
    """

    def __init__(self, chains, x0, sigma0=None):
        super(EmceeHammerMCMC, self).__init__(chains, x0, sigma0)

        # Need at least 3 chains
        if self._n_chains < 3:
            raise ValueError('Need at least 3 chains.')

        # Set initial state
        self._running = False

        # Current samples and log_pdfs
        self._current = None
        self._current_log_pdfs = None

        # Proposal:
        #  - n_chains points on the initial step
        #  - only a single point after
        #TODO: Update this class to algorithm 3
        self._proposed = None

        # Scale parameter (see docstring above)
        self._a = None
        self.set_scale(2)

    def ask(self):
        """ See :meth:`pints.MultiChainMCMC.ask()`. """
        # Initialise on first call (will set first proposal)
        if not self._running:
            self._initialise()

        # Propose new points
        if self._proposed is None:

            # Cycle through chains
            if len(self._remaining) == 0:
                self._remaining = np.arange(self._n_chains)
            self._k = self._remaining[0]
            self._remaining = np.delete(self._remaining, 0)

            j = self._random_select_other_index(self._k)
            x_j = self._current[j]
            x_k = self._current[self._k]

            self._z = self._sample_z(self._a)
            self._proposed = x_j + self._z * (x_k - x_j)

            # Ensure proposed is array containing a single sample
            #TODO Switch to algorithm 3
            self._proposed = np.array([self._proposed])
            self._proposed.setflags(write=False)

        # Return proposed points
        return self._proposed

    def _random_select_other_index(self, current_index):
        """
        Selects an index uniformly at random from all chains excluding the
        index of the current chain.
        """
        free_chains = list(range(self._n_chains))
        free_chains.remove(current_index)
        other_index = np.random.choice(free_chains)
        return other_index

    def _sample_z(self, a):
        """
        Samples ``z~g(z)`` where ``g(z)`` is proportional to ``1 / sqrt(z)``
        if ``z`` is in ``[1 / a, a]``; otherwise 0. It does this by using
        inverse transform sampling.
        """
        r = np.random.rand()
        return ((1 + r * (a - 1))**2) / a

    def _initialise(self):
        """
        Initialises the routine before the first iteration.
        """
        if self._running:
            raise RuntimeError('Already initialised.')

        # Propose x0 as first points
        # Note proposal is multiple points this time!
        self._current = None
        self._current_log_pdfs = None
        self._proposed = self._x0
        self._proposed.setflags(write=False)

        # Number of chains left to update in this cycle
        self._remaining = np.arange(self._n_chains)

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

        # Ensure fx is a numpy array
        proposed_log_pdf = np.array(fx, copy=True)

        # First points?
        if self._current is None:
            # Note that proposed and proposed_log_pdf are for all chains on
            # this iteration
            if not np.all(np.isfinite(proposed_log_pdf)):
                raise ValueError(
                    'Initial points for MCMC must have finite logpdf.')

            # Accept
            self._current = self._proposed  # already read-only
            self._current_log_pdfs = proposed_log_pdf
            self._current_log_pdfs.setflags(write=False)

            # Clear proposal
            self._proposed = None

            # Return first samples for chains
            accepted = np.array([True] * self._n_chains)
            return self._current, self._current_log_pdfs, accepted

        # Perform iteration, updating the selected chain
        # Note that proposed/proposed_log_pdf are length 1 here
        accepted = np.array([False] * self._n_chains)
        r_log = np.log(np.random.rand())
        log_q = (
            (self._n_parameters - 1) * np.log(self._z)
            + proposed_log_pdf[0] - self._current_log_pdfs[self._k])
        if log_q >= r_log:
            next = np.copy(self._current)
            next_log_pdfs = np.copy(self._current_log_pdfs)
            next[self._k] = self._proposed
            next_log_pdfs[self._k] = proposed_log_pdf
            self._current.setflags(write=False)
            self._current_log_pdfs.setflags(write=False)
            self._current = next
            self._current_log_pdfs = next_log_pdfs
            accepted[self._k] = True

        # Clear proposal
        self._proposed = None

        # Return samples to add to chains
        return self._current, self._current_log_pdfs, accepted

    def scale(self):
        """
        Returns the scale coefficient ``a`` used in updating the position of
        the chains.
        """
        return self._a

    def set_scale(self, scale):
        """
        Sets the scale coefficient ``a`` used in updating the position of the
        chains.
        """
        scale = float(scale)
        if scale <= 1:
            raise ValueError('The scale parameter must exceed 1.')
        self._a = scale

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 1

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[scale]``.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_scale(x[0])
