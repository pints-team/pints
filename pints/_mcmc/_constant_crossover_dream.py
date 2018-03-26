#
# Dream MCMC
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


class DreamMCMC(pints.MultiChainMCMC):
    """
    *Extends:* :class:`MultiChainMCMC`

    Uses differential evolution adaptive Metropolis (DREAM) MCMC as described
    in [1] to do posterior sampling from the posterior.

    In each step of the algorithm N chains are evolved using the following
    steps:

    1. Select proposal::

        x_proposed = x[i,r] + (1 + e) * gamma(delta, d, p_g) *
                     sum_j=1^delta (X[i,r1[j]] - x[i,r2[j]])
                     + epsilon

    where [r1[j], r2[j]] are random chain indices chosen (without replacement)
    from the ``N`` available chains, which must not equal each other or ``i``,
    where ``i`` indicates the current time step;
    ``delta ~ uniform_discrete(1,D)`` determines the number of terms to include
    in the summation::

        e ~ U(-b*, b*) in d dimensions;
        gamma(delta, d, p_g) =
          if p_g < u1 ~ U(0,1):
            2.38 / sqrt(2 * delta * d)
          else:
            1

    ``epsilon ~ N(0,b)`` in ``d`` dimensions (where ``d`` is the dimensionality
    of the parameter vector).

    2. Modify random subsets of the proposal according to a crossover
    probability CR::

        for j in 1:N:
          if 1 - CR > u2 ~ U(0,1):
            x_proposed[j] = x[j]
          else:
            x_proposed[j] = x_proposed[j] from 1

    If ``x_proposed / x[i,r] > u ~ U(0,1)``, then
    ``x[i+1,r] = x_proposed``; otherwise, ``x[i+1,r] = x[i]``.

    [1] "Accelerating Markov Chain Monte Carlo Simulation by Differential
    Evolution with Self-Adaptive Randomized Subspace Sampling",
    2009, Vrugt et al.,
    International Journal of Nonlinear Sciences and Numerical Simulation.
    """
    def __init__(self, log_pdf, x0, sigma0=None):
        super(DreamMCMC, self).__init__(chains, x0, sigma0)

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

        # Normal proposal std.
        self._b = 0.01

        # b* distribution for e ~ U(-b*, b*)
        self._b_star = 0.01

        # Probability of longer gamma versus regular
        self._p_g = 0.2

        # Determines maximum delta to choose in sums
        self._D = 3

        # Constant crossover probability boolean
        self._constant_crossover = False

        # Crossover probability for variable CR case
        self._nCR = 3

        # Constant CR probability for constant CR probability
        self._CR = 0.5

    def ask(self):
        """ See :meth:`pints.MultiChainMCMC.ask()`. """
        # Initialise on first call
        if not self._running:
            self._initialise()

        # Propose new points
        if self._proposed is None:

            #TODO

            #self._proposed = np.zeros(self._current.shape)
            #for j in range(self._chains):
            #    r1, r2 = r_draw(j, self._chains)
            #    self._proposed[j] = (
            #        self._current[j]
            #        + self._gamma * (self._current[r1] - self._current[r2])
            #        + np.random.normal(0, self._b * self._mu, self._mu.shape)
            #    )

            # Set as read only
            self._proposed.setflags(write=False)

        # Return proposed points
        return self._proposed



    def run(self):
        """See: :meth:`pints.MCMC.run()`."""

        # Initial starting parameters
        mu = self._x0
        current = self._x0
        current_log_pdf = self._log_pdf(current)
        if not np.isfinite(current_log_pdf):
            raise ValueError(
                'Suggested starting position has a non-finite log-pdf.')

        # chains of stored samples
        chains = np.zeros(
            (self._iterations, self._num_chains, self._dimension))
        current_log_pdf = np.zeros(self._num_chains)

        # Set initial values
        for j in range(self._num_chains):
            chains[0, j, :] = np.random.normal(
                loc=mu, scale=mu / 100, size=len(mu))
            current_log_pdf[j] = self._log_pdf(chains[0, j, :])





        # Go!
        for i in range(1, self._iterations):
            for j in range(self._num_chains):
                # Step 1. Select (initial) proposal
                delta = int(np.random.choice(self._D, 1)[0] + 1)
                dX = 0
                u1 = np.random.rand()
                if self._p_g < u1:
                    gamma = 2.38 / np.sqrt(2 * delta * self._dimension)
                else:
                    gamma = 1.0
                e = np.random.uniform(low=-self._b_star * mu,
                                      high=self._b_star * mu)
                for k in range(0, delta):
                    r1, r2 = r_draw(j, self._num_chains)
                    dX += (1 + e) * gamma * (chains[i - 1, r1, :] -
                                             chains[i - 1, r2, :])
                proposed = (
                    chains[i - 1, j, :] + dX
                    + np.random.normal(loc=0, scale=self._b * mu, size=len(mu))

                # Step 2. Randomly set elements of proposal to original
                for d in range(0, self._dimension):
                    u2 = np.random.rand()
                    if 1 - self._CR > u2:
                        proposed[d] = chains[i - 1, j, d]

                # Accept/reject
                u = np.log(np.random.rand())
                proposed_log_pdf = self._log_pdf(proposed)

                if u < proposed_log_pdf - current_log_pdf[j]:
                    chains[i, j, :] = proposed
                    current_log_pdf[j] = proposed_log_pdf
                else:
                    chains[i, j, :] = chains[i - 1, j, :]



    def set_b(self, b):
        """
        Sets the normal scale coefficient used in updating the position of each
        chain.
        """
        if b < 0:
            raise ValueError('normal scale coefficient must be non-negative.')
        self._b = b

    def set_gamma(self, gamma):
        """
        Sets the gamma coefficient used in updating the position of each
        chain.
        """
        if gamma < 0:
            raise ValueError('Gamma must be non-negative.')
        self._gamma = gamma




def r_draw(i, num_chains):
    #TODO: Needs a docstring!
    r1, r2 = np.random.choice(num_chains, 2, replace=False)
    while(r1 == i or r2 == i or r1 == r2):
        r1, r2 = np.random.choice(num_chains, 2, replace=False)
    return r1, r2
