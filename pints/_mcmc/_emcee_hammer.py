#
# Emcee hammer MCMC
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
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

    - Set ``q = z^{N - 1} p(Y) / p(X_k(t))``.

    - Sample ``r ~ U(0, 1)``.

    - If ``r <= q``, set ``X_k(t + 1)`` equal to ``Y``, if not use ``X_k(t)``.

    Here, ``N`` is the number of chains (or walkers), and ``g(z)`` is
    proportional to ``1 / sqrt(z)`` if ``z`` is in  ``[1 / a, a]`` or to 0,
    otherwise (where ``a`` is a parameter with default value ``2``).

    References
    ----------
    .. [1] "emcee: The MCMC Hammer", Daniel Foreman-Mackey, David W. Hogg,
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

        # Current samples and proposed log_pdfs
        self._current = None
        self._current_log_pdfs = None

        # Single proposed point!
        #TODO: Update this class to algorithm 3
        self._proposed = None

        # Scale parameter (see docstring above)
        self._a = None
        self.set_scale(2.0)

    def ask(self):
        """ See :meth:`pints.MultiChainMCMC.ask()`. """
        # Initialise on first call (will set first proposal)
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

            # Ensure proposed is array of samples (with length 1)
            #TODO Switch to algorithm 3
            self._proposed = np.array([self._proposed])

        # Set as read only
        self._proposed.setflags(write=False)

        # Return proposed points
        return self._proposed

    def current_log_pdfs(self):
        """ See :meth:`MultiChainMCMC.current_log_pdfs()`. """
        return self._current_log_pdfs

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

        # Ensure proposed_log_pdf is a numpy array
        proposed_log_pdf = np.array(fx, copy=True)

        # First points?
        if self._current is None:
            if not np.all(np.isfinite(proposed_log_pdf)):
                raise ValueError(
                    'Initial points for MCMC must have finite logpdf.')

            # Accept
            # NOTE: FIRST STEP PROPOSED IS MULTIPLE POINTS
            self._current = self._proposed
            self._current_log_pdfs = proposed_log_pdf
            self._current_log_pdfs.setflags(write=False)

            # Clear proposal
            self._proposed = None

            # Return first samples for chains
            return self._current

        # Perform iteration
        next = np.array(self._current, copy=True)
        next_log_pdfs = np.array(self._current_log_pdfs, copy=True)

        # Update the selected chain
        #TODO Switch to algorithm 3, doing 2 sets of chains per iteration
        r_log = np.log(np.random.rand())
        q = (
            (self._chains - 1) * np.log(self._z)
            + proposed_log_pdf - self._current_log_pdfs[self._k])
        if q >= r_log:
            next[self._k] = self._proposed[0]
            next_log_pdfs[self._k] = proposed_log_pdf
            self._current = next
            self._current_log_pdfs = next_log_pdfs
            self._current_log_pdfs.setflags(write=False)

        # Clear proposal
        self._proposed = None

        # Return samples to add to chains
        self._current.setflags(write=False)
        return self._current

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
        if scale <= 0:
            raise ValueError('The scale parameter must be positive.')
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
