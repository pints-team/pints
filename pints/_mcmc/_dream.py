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
        self._delta_max = 3

        # 1 / Variable crossover probability
        self._nCR = 3

        #
        # TODO: WARM UP PERIOD
        #
        self._in_warm_up = False

    def ask(self):
        """ See :meth:`pints.MultiChainMCMC.ask()`. """
        # Initialise on first call
        if not self._running:
            self._initialise()

        # Propose new points
        # Note: Initialise sets the proposal for the very first step
        if self._proposed is None:

            self._proposed = np.array(self._current, copy=True)

            for j in range(self._chains):

                # Select initial proposal for chain j
                delta = int(np.random.choice(self._D, 1)[0] + 1)
                if self._p_g < np.random.rand():
                    gamma = 2.38 / np.sqrt(2 * delta * self._n_parameters)
                else:
                    gamma = 1.0

                e = np.random.uniform(
                    low=-self._b_star * mu, high=self._b_star * mu)

                dX = 0
                for k in range(0, delta):
                    r1, r2 = r_draw(j, self._chains)
                    dX += (1 + e) * gamma * (
                        self._current[r1] - self._current[r2])

                self._proposed[j] += dX + np.random.normal(
                    loc=0, scale=self._b * mu, size=self._n_parameters)

                # Select CR from multinomial distribution
                m = np.nonzero(np.random.multinomial(self._nCR, self._p))[0][0]
                CR = (m + 1) / self._nCR
                self._L[m] += 1

                # Randomly set elements of proposal to back original
                for d in range(0, self._dimension):
                    if 1 - CR > np.random.rand():
                        self._proposed[j][d] = self._current[j][d]

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
        self._proposed = np.array(self._x0, copy=True)

        # Set proposal as read-only
        self._proposed.setflags(write=False)

        #TODO: Would prefer not to use this method, have user specified x0 used instead
        # Set initial values
        #for j in range(self._num_chains):
        #    chains[0, j, :] = np.random.normal(loc=mu, scale=mu / 100.0,
        #                                       size=len(mu))
        #    current_log_likelihood[j] = self._log_likelihood(chains[0, j, :])

        # Set mu
        self._mu = np.mean(self._x0, axis=0)

        # Set p, L and Delta
        self._p = np.repeat(1 / self._nCR, self._nCR)
        self._L = np.zeros(self._nCR)
        self._Delta = np.zeros(self._nCR)


        # Update sampler state
        self._running = True



        #
        #
        # RUN RUN RUN RUN RUN
        #
        #TODO self._current isn't used, method uses last from chain!


        # chains of stored samples
        #chains = np.zeros((self._iterations, self._num_chains, self._dimension))
        #current_log_likelihood = np.zeros(self._num_chains)

    def _log_init(self, logger):
        """ See :meth:`Loggable._log_init()`. """
        #logger.add_float('Accept.')
        #TODO

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        #logger.log(self._acceptance)
        #TODO

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'DiffeRential Evolution Adaptive Metropolis (DREAM) MCMC'

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

        # Warm-up? Then store current as 'previous' positions
        if self._in_warm_up:
            previous = np.array(self._current, copy=True)

        # Accept/reject for each chain
        u = np.log(np.random.rand(self._chains))
        for j in range(self._chains):
            if u[j] < proposed_log_pdfs[j] - self._current_log_pdfs[j]:
                self._current[j] = self._proposed[j]
                self._current_log_pdfs[j] = proposed_log_pdfs[j]

        # Warm-up? The update CR distribution based on (new) current & previous
        if self._in_warm_up:

            # Update CR distribution
            for j in range(self._chains):
                for d in range(0, self._n_parameters):
                    delta = self._current[j] - previous[j]
                    #TODO: UH OH!!!!
                    self._Delta[m] += (delta)**2 / np.var(chains[:, j, d])

            for k in range(0, self._nCR):
                p[k] = i * self._num_chains * (Delta[k] / float(L[k])) / np.sum(Delta)  # NOQA
            p = p / np.sum(p)

            #TODO: RETURN
            return 123


        # Post warm-up





    def b(self):
        """
        Returns the normal scale coefficient used in updating the position of
        each chain.
        """
        return self._b

    def set_b(self, b):
        """
        Sets the normal scale coefficient used in updating the position of each
        chain.
        """
        if b < 0:
            raise ValueError('normal scale coefficient must be non-negative.')
        self._b = b




def r_draw(i, num_chains):
    #TODO: Needs a docstring!
    r1, r2 = np.random.choice(num_chains, 2, replace=False)
    while(r1 == i or r2 == i or r1 == r2):
        r1, r2 = np.random.choice(num_chains, 2, replace=False)
    return r1, r2
