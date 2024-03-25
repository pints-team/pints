#
# Bare-bones re-implementation of CMA-ES.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
import pints
import warnings

from numpy.linalg import norm
from scipy.special import gamma


class BareCMAES(pints.PopulationBasedOptimiser):
    """
    Finds the best parameters using the CMA-ES method described in [1, 2],
    using a bare bones re-implementation.

    For general use, we recommend the :class:`pints.CMAES` optimiser, which
    wraps around the ``cma`` module provided by the authors of CMA-ES. The
    ``cma`` module provides a battle-tested version of the optimiser.

    The role of this class, is to provide a simpler implementation of only the
    core algorithm of CMA-ES, which is easier to read and analyse, and which
    can be used to compare with bare implementations of other methods.

    Extends :class:`PopulationBasedOptimiser`.

    References
    ----------
    .. [1] The CMA Evolution Strategy: A Tutorial
           Nikolaus Hanse, arxiv
           https://arxiv.org/abs/1604.00772

    .. [2] Hansen, Mueller, Koumoutsakos (2003) "Reducing the time complexity
           of the derandomized evolution strategy with covariance matrix
           adaptation (CMA-ES)". Evolutionary Computation
           https://doi.org/10.1162/106365603321828970

    """

    def __init__(self, x0, sigma0=0.1, boundaries=None):
        super(BareCMAES, self).__init__(x0, sigma0, boundaries)

        # Set initial state
        self._running = False
        self._ready_for_tell = False

        # Best solution found
        self._x_best = pints.vector(x0)
        self._f_best = np.inf

        # Number of iterations run
        self._iterations = 0

        # Mean of the proposal distribution
        self._mu = np.copy(self._x0)

        # Approximate value at self._mu
        self._f_guessed = np.inf

        # Step size
        self._eta = np.min(self._sigma0)

        # Covariance matrix C and decomposition in rotation R and scaling S
        # A decomposition C = R S S R.T can be made, such that R is the matrix
        # of eigenvectors of C, and S is a diagonal matrix containing the
        # square roots of the eigenvalues of C.
        # Here, R and S can be interpreted as a rotation and a scaling matrix
        # respectively.
        # Note that only C is updated directly, while R and S are simply
        # recalculated at every step.
        self._C = np.identity(self._n_parameters)
        self._R = np.identity(self._n_parameters)
        self._S = np.identity(self._n_parameters)

        # Constant used in tell()
        self._e = (
            np.sqrt(2)
            * gamma((self._n_parameters + 1) / 2)
            / gamma(self._n_parameters / 2)
        )

    def ask(self):
        """ See :meth:`Optimiser.ask()`. """
        # Initialise on first call
        if not self._running:
            self._initialise()

        # Ready for tell now
        self._ready_for_tell = True

        # Create new samples
        # Normalised samples: centered at zero and no rotation or scaling
        self._zs = np.array([np.random.normal(0, 1, self._n_parameters)
                             for i in range(self._population_size)])

        # Centered samples: centered at zero, with rotation and scaling
        self._ys = np.array([self._R.dot(self._S).dot(z) for z in self._zs])

        # Samples from N(mu, eta**2 * C)
        self._xs = np.array([self._mu + self._eta * y for y in self._ys])

        # Boundaries? Then only pass user xs that are within bounds
        if self._boundaries is not None:
            self._user_ids = np.nonzero(
                [self._boundaries.check(x) for x in self._xs])
            self._user_xs = self._xs[self._user_ids]
            if len(self._user_xs) == 0:     # pragma: no cover
                warnings.warn('All points requested by BareCMAES are outside'
                              ' the boundaries.')
        else:
            self._user_xs = self._xs

        # Set as read-only and return
        self._user_xs.setflags(write=False)
        return self._user_xs

    def cov(self, decomposed=False):
        """
        Returns the current covariance matrix ``C`` of the proposal
        distribution.

        If the optional argument ``decomposed`` is set to ``True``, a tuple
        ``(R, S)`` will be returned such that ``R`` contains the eigenvectors
        of ``C`` while ``S`` is a diagonal matrix containing the squares of the
        eigenvalues of ``C``, such that ``C = R S S R.T``.
        """
        if decomposed:
            return self._R, self._S
        else:
            return np.copy(self._C)

    def f_best(self):
        """ See :meth:`Optimiser.f_best()`. """
        return self._f_best

    def f_guessed(self):
        """ See :meth:`Optimiser.f_guessed()`. """
        return self._f_guessed

    def mean(self):
        """
        Returns the current mean of the proposal distribution.
        """
        return self.x_guessed()

    def _initialise(self):
        """
        Initialises the optimiser for the first iteration.
        """
        assert (not self._running)

        # Parent generation population size
        # The parameter parent_pop_size is the mu in the papers. It represents
        # the size of a parent population used to update our paramters.
        self._parent_pop_size = self._population_size // 2

        # Weights, all set equal for the moment
        # Sum of all positive weights should be 1
        self._W = 1 + np.arange(self._population_size)
        self._W = np.log(0.5 * (self._population_size + 1)) - np.log(self._W)

        # Inverse of the sum of the first parent weights squared (variance
        # effective selection mass)
        self._muEff = (
            np.sum(self._W[:self._parent_pop_size]) ** 2
            / np.sum(np.square(self._W[:self._parent_pop_size]))
        )

        # Inverse of the Sum of the last weights squared (variance effective
        # selection mass)
        self._muEffMinus = (
            np.sum(self._W[self._parent_pop_size:]) ** 2
            / np.sum(np.square(self._W[self._parent_pop_size:]))
        )

        # cummulation, evolution paths, used to update Cov matrix and sigma)
        self._pc = np.zeros(self._n_parameters)
        self._psig = np.zeros(self._n_parameters)

        # learning rate for the mean
        self._cm = 1

        # Decay rate of the evolution path for C
        self._ccov = (4 + self._muEff / self._n_parameters) / (
            self._n_parameters + 4 + 2 * self._muEff / self._n_parameters)

        # Decay rate of the evolution path for sigma
        self._csig = (2 + self._muEff) / (self._n_parameters + 5 + self._muEff)

        # See rank-1 vs rank-mu updates
        # Learning rate for rank-1 update
        self._c1 = 2 / ((self._n_parameters + 1.3) ** 2 + self._muEff)

        # Learning rate for rank-mu update
        self._cmu = min(
            2 * (self._muEff - 2 + 1 / self._muEff)
            / ((self._n_parameters + 2) ** 2 + self._muEff),
            1 - self._c1
        )

        # Damping of the step-size (sigma0) update
        self._dsig = 1 + self._csig + 2 * max(
            0, np.sqrt((self._muEff - 1) / (self._n_parameters + 1)) - 1)

        # Parameters from the Table 1 of [1]
        alpha_mu = 1 + self._c1 / self._cmu
        alpha_mueff = 1 + 2 * self._muEffMinus / (self._muEff + 2)
        alpha_pos_def = \
            (1 - self._c1 - self._cmu) / (self._n_parameters * self._cmu)

        # Rescale the weights
        sum_pos = np.sum(self._W[self._W > 0])
        sum_neg = np.sum(self._W[self._W < 0])
        scale_pos = 1 / sum_pos
        scale_neg = min(alpha_mu, alpha_mueff, alpha_pos_def) / -sum_neg
        self._W[self._W > 0] *= scale_pos
        self._W[self._W < 0] *= scale_neg

        # Update optimiser state
        self._running = True

    def name(self):
        """ See :meth:`Optimiser.name()`. """
        return 'Bare-bones CMA-ES'

    def running(self):
        """ See :meth:`Optimiser.running()`. """
        return self._running

    def stop(self):
        """ See :meth:`Optimiser.stop()`. """

        # We use the condition number defined in the pycma code at
        # cma/evolution_strategy.py#L2965.
        cond = np.diagonal(self._S)
        cond = (np.max(cond) / np.min(cond)) ** 2
        if cond > 1e14:     # pragma: no cover
            return 'Ill-conditioned covariance matrix'
        return False

    def _suggested_population_size(self):
        """
        See :meth:`PopulationBasedOptimiser._suggested_population_size().
        """
        return 4 + int(3 * np.log(self._n_parameters))

    def tell(self, fx):
        """ See :meth:`Optimiser.tell()`. """

        # Check ask-tell pattern
        if not self._ready_for_tell:
            raise Exception('ask() not called before tell()')
        self._ready_for_tell = False

        # Update iteration count
        self._iterations += 1

        # Some aliases for readability
        n = self._n_parameters
        npo = self._population_size
        npa = self._parent_pop_size

        # Boundaries? Then reconstruct full fx vector
        if self._boundaries is not None and len(fx) < npo:
            user_fx = fx
            fx = np.ones((npo, )) * np.inf
            fx[self._user_ids] = user_fx

        # Order the points from best to worst score
        order = np.argsort(fx)
        xs = self._xs[order]
        zs = self._zs[order]
        ys = self._ys[order]

        # Update the mean
        self._mu += self._cm * np.sum(
            ((xs[:npa] - self._mu).T * self._W[:npa]).T, axis=0)

        # Get the weighted means of y and z
        zmeans = np.sum((zs[:npa].T * self._W[:npa]).T, 0)
        ymeans = np.sum((ys[:npa].T * self._W[:npa]).T, 0)

        # Evolution path of sigma (the step size)
        # Note that self._R.dot(zmeans) =
        #     self._R.dot(np.linalg.inv(self._S)).dot(self._R.T).dot(ymeans)
        # Normalizing constants for the evolution path udpate
        c = np.sqrt(self._csig * (2 - self._csig) * self._muEff)
        self._psig = (1 - self._csig) * self._psig + c * self._R.dot(zmeans)

        # In cma/sigma_adaptation.py#L71 they are NOT using exp_size_N0I, but
        # instead use a term based on n_parameters.
        # Heaviside function helps to stall the of pc if norm ||psig|| is too
        # large. This helps to prevent too fast increases in the axes of C when
        # the step size is too small.
        cond = (
            norm(self._psig)
            / np.sqrt(1 - (1 - self._csig) ** (2 * (self._iterations + 1)))
            < (1.4 + 2 / (n + 1)) * self._e
        )
        h_sig = 1 if cond else 0
        delta_sig = (1 - h_sig) * self._ccov * (2 - self._ccov)

        # Evolution path for the rank-1 update
        c = np.sqrt(self._ccov * (2 - self._ccov) * self._muEff)
        self._pc = (1 - self._ccov) * self._pc + h_sig * c * ymeans

        # Weight changes taken from the tutorial (no explanation is given for
        # the change) these weights are used for the rank mu update only.
        # They allow to keep positive definiteness according to
        # cma/purecma.py#L419
        temp_weights = np.copy(self._W)
        for i, w in enumerate(self._W):
            if w < 0:
                temp_weights[i] = w * n / (norm(self._R * zs[i]) ** 2)

        # Update the Covariance matrix:

        # Calculate the rank 1 update using the evolution path
        rank1 = self._c1 * np.outer(self._pc, self._pc)

        # Calculate the rank-mu update
        yy = np.array([np.outer(y, y) for y in ys]).T
        rankmu = self._cmu * np.sum((yy * temp_weights).T, 0)

        # Update C
        self._C = rank1 + rankmu + self._C * (
            1 + delta_sig * self._c1 - self._c1 - self._cmu * sum(self._W))

        # Avoid numerical issues by forcing C to be symmetric
        self._C = np.triu(self._C) + np.triu(self._C, 1).T

        # Update the step size
        # Here we are simply looking at the ratio of the length of the
        #  evolution path vs its expected lenght ( E|Gaussian(0,I)|)
        # We use the difference of the ratio with 1 and scale using the
        #  learning rate and the damping parameters.
        self._eta *= np.exp(
            self._csig / self._dsig * (norm(self._psig) / self._e - 1))

        # Update eigenvectors and eigenvalues of C
        eig = np.linalg.eigh(self._C)
        self._S = np.sqrt(np.diag(eig[0]))
        self._R = eig[1]

        # Update f_guessed on the assumption that the lowest value in our
        # sample approximates f(mu)
        self._f_guessed = fx[order[0]]

        # Update x_best and f_best
        if self._f_guessed < self._f_best:
            self._f_best = self._f_guessed
            self._x_best = np.array(xs[0], copy=True)

    def x_best(self):
        """ See :meth:`Optimiser.x_best()`. """
        return self._x_best

    def x_guessed(self):
        """ See :meth:`Optimiser.x_guessed()`. """
        return np.array(self._mu, copy=True)
