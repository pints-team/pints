#
# Dream MCMC
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import numpy as np


class DreamMCMC(pints.MultiChainMCMC):
    """
    Uses differential evolution adaptive Metropolis (DREAM) MCMC as described
    in [1]_ to perform posterior sampling from the posterior.

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

    Here b > 0, b* > 0,  1 >= p_g >= 0, 1 >= CR >= 0.

    Extends :class:`MultiChainMCMC`.

    References
    ----------
    .. [1] "Accelerating Markov Chain Monte Carlo Simulation by Differential
           Evolution with Self-Adaptive Randomized Subspace Sampling",
           2009, Vrugt et al., International Journal of Nonlinear Sciences and
           Numerical Simulation.
           https://doi.org/10.1515/IJNSNS.2009.10.3.273
    """

    def __init__(self, chains, x0, sigma0=None):
        super(DreamMCMC, self).__init__(chains, x0, sigma0)

        # Need at least 3 chains
        if self._n_chains < 3:
            raise ValueError('Need at least 3 chains.')

        # Set initial state
        self._running = False

        # Current points and proposed points
        self._current = None
        self._current_log_pdfs = None
        self._proposed = None

        #
        # Default settings
        #

        # Gaussian proposal std.
        self._b = 0.01

        # b* distribution for e ~ U(-b*, b*)
        self._b_star = 0.01

        # Probability of higher gamma versus regular
        self._p_g = 0.2

        # Determines maximum delta to choose in sums
        self._delta_max = None
        self.set_delta_max(min(3, self._n_chains - 2))

        # Initial phase
        self._initial_phase = True

        # Variable or constant crossover mode
        self._constant_crossover = False

        # Constant CR probability
        self._CR = 0.5

        # Since of multinomial crossover dist for variable CR prob
        self._nCR = 3

    def ask(self):
        """ See :meth:`pints.MultiChainMCMC.ask()`. """
        # Initialise on first call
        if not self._running:
            self._initialise()

        # Propose new points
        # Note: Initialise sets the proposal for the very first step
        if self._proposed is None:

            self._proposed = np.array(self._current, copy=True)

            for j in range(self._n_chains):

                # Select initial proposal for chain j
                delta = int(np.random.choice(self._delta_max, 1)[0] + 1)
                if self._p_g < np.random.rand():
                    gamma = 2.38 / np.sqrt(2 * delta * self._n_parameters)
                else:
                    gamma = 1.0

                e = self._b_star * self._mu
                e = np.random.uniform(-e, e)

                dX = 0
                for k in range(0, delta):
                    r1, r2 = self._draw(j)
                    dX += (1 + e) * gamma * (
                        self._current[r1] - self._current[r2])

                self._proposed[j] += dX + np.random.normal(
                    loc=0, scale=np.abs(self._b * self._mu),
                    size=self._n_parameters)

                # Set crossover probability
                if self._constant_crossover:
                    CR = self._CR
                else:
                    # Select CR from multinomial distribution
                    self._m[j] = np.nonzero(
                        np.random.multinomial(self._nCR, self._p))[0][0]
                    CR = (self._m[j] + 1) / self._nCR
                    self._L[self._m[j]] += 1

                # Randomly set elements of proposal to back original
                for d in range(self._n_parameters):
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

        # Set mu
        self._mu = np.mean(self._x0, axis=0)

        # Set initial p, L and Delta
        self._p = np.repeat(1.0 / self._nCR, self._nCR)
        self._L = np.zeros(self._nCR)
        self._delta = np.zeros(self._nCR)

        # Create empty array of m indices
        self._m = [0] * self._n_chains

        # Iteration tracking for running variance
        # See: https://www.johndcook.com/blog/standard_deviation/
        # Algorithm based on Knuth TAOCP vol 2, 3rd edition, page 232
        self._iterations = 0
        self._varm = None
        self._vars = None
        self._variance = None

        # Update sampler state
        self._running = True

    def in_initial_phase(self):
        """ See :meth:`pints.MCMCSampler.in_initial_phase()`. """
        return self._initial_phase

    def _log_init(self, logger):
        """ See :meth:`Loggable._log_init()`. """
        # logger.add_float('Accept.')
        # TODO

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        # logger.log(self._acceptance)
        # TODO

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'DiffeRential Evolution Adaptive Metropolis (DREAM) MCMC'

    def needs_initial_phase(self):
        """ See :meth:`pints.MCMCSampler.needs_initial_phase()`. """
        return True

    def set_initial_phase(self, initial_phase):
        """ See :meth:`pints.MCMCSampler.needs_initial_phase()`. """
        self._initial_phase = bool(initial_phase)

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
            self._current_log_pdfs = np.copy(proposed_log_pdfs)
            self._current_log_pdfs.setflags(write=False)
            accepted = np.array([True] * self._n_chains)

            # Clear proposal
            self._proposed = None

            # Return first samples for chains
            return self._current, self._current_log_pdfs, accepted

        # Perform iteration
        next = np.copy(self._current)
        next_log_pdfs = np.copy(self._current_log_pdfs)

        # Sample uniform numbers
        u = np.log(np.random.uniform(size=self._n_chains))

        # Get chains to be updated
        i = u < (proposed_log_pdfs - self._current_log_pdfs)

        # Update (part 1)
        next[i] = self._proposed[i]
        next_log_pdfs[i] = proposed_log_pdfs[i]

        # Warm-up? Then update CR distribution based on current & previous
        if self._initial_phase and not self._constant_crossover:

            # Update running mean and variance
            if self._iterations == 0:
                self._varm = self._current
                self._variance = self._vars = self._current * 0
            else:
                new_varm = self._varm + (self._current - self._varm) / (
                    self._iterations + 1)
                self._vars += (self._current - self._varm) * (
                    self._current - new_varm)
                self._varm = new_varm
                self._variance = self._vars / (self._iterations + 1)

                # Update CR distribution
                delta = (next - self._current)**2
                for j in range(self._n_chains):
                    for d in range(0, self._n_parameters):
                        self._delta[self._m[j]] += (
                            delta[j][d] / max(self._variance[j][d], 1e-11))

                self._p = self._iterations * self._n_chains * self._delta
                d1 = self._L * np.sum(self._delta)
                d1[d1 == 0] += 1e-11
                self._p /= d1
                d2 = max(np.sum(self._p), 1e-11)
                self._p /= d2

            # Update iteration count for running mean/variance
            self._iterations += 1

        # Update (part 2)
        self._current = next
        self._current_log_pdfs = next_log_pdfs
        self._current.setflags(write=False)
        self._current_log_pdfs.setflags(write=False)

        # Clear proposal
        self._proposed = None

        # Return samples to add to chains
        return self._current, self._current_log_pdfs, i

    def b(self):
        """
        Returns the Gaussian scale coefficient used in updating the position of
        each chain.
        """
        return self._b

    def b_star(self):
        """
        Returns b*, which determines the weight given to other chains'
        positions in determining new positions (see :meth:`set_b_star()`).
        """
        return self._b_star

    def constant_crossover(self):
        """
        Returns ``True`` if constant crossover mode is enabled.
        """
        return self._constant_crossover

    def CR(self):
        """
        Returns the probability of crossover occurring if constant crossover
        mode is enabled (see :meth:`set_CR()`).
        """
        return self._CR

    def delta_max(self):
        """
        Returns the maximum number of other chains' positions to use to
        determine the next sampler position (see :meth:`set_delta_max()`).
        """
        return self._delta_max

    def _draw(self, i):
        """
        Select 2 random chains, not including chain i.
        """
        r1, r2 = np.random.choice(self._n_chains, 2, replace=False)
        while r1 == i or r2 == i or r1 == r2:
            r1, r2 = np.random.choice(self._n_chains, 2, replace=False)
        return r1, r2

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 8

    def nCR(self):
        """
        Returns the size of the discrete crossover probability distribution
        (only used if constant crossover mode is disabled), see
        :meth:`set_nCR()`.
        """
        return self._nCR

    def p_g(self):
        """
        Returns ``p_g``. See :meth:`set_p_g()`.
        """
        return self._p_g

    def set_b(self, b):
        """
        Sets the Gaussian scale coefficient used in updating the position of
        each chain (must be non-negative).
        """
        if b < 0:
            raise ValueError(
                'Gaussian scale coefficient must be non-negative.')
        self._b = b

    def set_constant_crossover(self, enabled):
        """
        Enables/disables constant-crossover mode (must be bool).
        """
        self._constant_crossover = True if enabled else False

    def set_b_star(self, b_star):
        """
        Sets b*, which determines the weight given to other chains' positions
        in determining new positions (must be non-negative).
        """
        if b_star < 0:
            raise ValueError('b* must be non-negative.')
        self._b_star = b_star

    def set_p_g(self, p_g):
        """
        Sets ``p_g`` which is the probability of choosing a higher ``gamma``
        versus regular (a higher ``gamma`` means that other chains are given
        more weight). ``p_g`` must be in the range [0, 1].
        """
        if p_g < 0 or p_g > 1:
            raise ValueError('p_g must be in the range [0, 1].')
        self._p_g = p_g

    def set_delta_max(self, delta_max):
        """
        Sets the maximum number of other chains' positions to use to determine
        the next sampler position. ``delta_max`` must be in the range
        ``[1, nchains - 2]``.
        """
        delta_max = int(delta_max)
        if delta_max > (self._n_chains - 2):
            raise ValueError(
                'delta_max must be less than or equal to the number of chains '
                'minus 2.')
        if delta_max < 1:
            raise ValueError('delta_max must be at least 1.')
        self._delta_max = delta_max

    def set_CR(self, CR):
        """
        Sets the probability of crossover occurring if constant crossover mode
        is enabled. CR is a probability and so must be in the range ``[0, 1]``.
        """
        if CR < 0 or CR > 1:
            raise ValueError('CR is a probability and so must be in [0, 1].')
        self._CR = CR

    def set_nCR(self, nCR):
        """
        Sets the size of the discrete crossover probability distribution (only
        used if constant crossover mode is disabled). ``nCR`` must be greater
        than or equal to 2.
        """
        if nCR < 2:
            raise ValueError(
                'Length of discrete crossover distribution must be 2 or'
                ' greater.')
        self._nCR = int(nCR)

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[b, b_star, p_g, delta_max,
        initial_phase, constant_crossover, CR, nCR]``.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_b(x[0])
        self.set_b_star(x[1])
        self.set_p_g(x[2])
        self.set_delta_max(x[3])
        self.set_initial_phase(x[4])
        self.set_constant_crossover(x[5])
        self.set_CR(x[6])
        self.set_nCR(x[7])
