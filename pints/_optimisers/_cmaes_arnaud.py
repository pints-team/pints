#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#

from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import logging
import numpy as np
import math
import sys
import pints


class CMAES(pints.PopulationBasedOptimiser):
    """
    Finds the best parameters using the CMA-ES method described in [1, 2].

    CMA-ES stands for Covariance Matrix Adaptation Evolution Strategy, and is
    designed for non-linear derivative-free optimization problems.

    *Extends:* :class:`PopulationBasedOptimiser`

    [1] https://arxiv.org/pdf/1604.00772.pdf

    [2] Hansen, Mueller, Koumoutsakos (2006) Reducing the time complexity of
    the derandomized evolution strategy with covariance matrix adaptation
    (CMA-ES).

    Important note: The parameter parent_pop_size is the mu in the papers. It represents the size of a parent population
        used to update our paramters.

    """

    def __init__(self, x0, sigma0=0.1, boundaries=None):
        super(CMAES, self).__init__(x0, sigma0, boundaries)

        # Set initial state
        self._running = False
        self._ready_for_tell = False

        # Best solution found
        self._xbest = pints.vector(x0)
        self._fbest = float('inf')

        # Python logger
        self._logger = logging.getLogger(__name__)

        self._counter = 0

    def ask(self):
        """ See :meth:`Optimiser.ask()`. """
        # Initialise on first call
        if not self._running:
            self._initialise()

        # Ready for tell now
        self._ready_for_tell = True

        # Create new samples
        self._zs = np.array([np.random.multivariate_normal([0] * self._n_parameters, np.identity(self._n_parameters))
                             for _ in range(self._population_size)])

        self._ys = np.array([self._B.dot(self._D).dot(z) for z in self._zs])

        self._xs = np.array([self._x0 + self._sigma0 * self._ys[i]
                             for i in range(self._population_size)])

        if self._manual_boundaries:
            if isinstance(self._boundaries, pints.RectangularBoundaries):
                upper = self._boundaries.upper()
                lower = self._boundaries.lower()
            self._user_xs = [[x[i] % upper[i] + lower[i] for i in range(len(x))] for x in self._xs]
        else:
            self._user_xs = self._xs

        self._user_xs = np.array(self._user_xs)
        # Set as read-only and return
        self._user_xs.setflags(write=False)
        return self._user_xs

    def fbest(self):
        """ See :meth:`Optimiser.fbest()`. """
        if not self._running:
            return float('inf')
        return self._fbest

    def _initialise(self):
        """
        Initialises the optimiser for the first iteration.
        """
        assert (not self._running)

        # Set boundaries, or use manual boundary checking
        self._manual_boundaries = False
        self._boundary_transform = None
        if self._boundaries is not None:
            self._manual_boundaries = True

        # CMA-ES needs a single standard deviation as input, use the smallest
        # in the vector (if the user passed in a scalar, this will be the
        # value used). THIS IS ALSO THE STEP SIZE
        self._sigma0 = np.min(self._sigma0)

        # Eigenvectors
        self._B = np.identity(self._n_parameters)
        # SquareRoot of Diagonal of EigenValues
        self._D = np.identity(self._n_parameters)
        # Cov-matrix (also identity)
        self._C = self._B.dot(self._D).dot(self._D.T).dot(self._B.T)

        # Parent generation population size
        # Not sure if the limitation to dim is good
        # This limitation is supposed to prevent a mistake in the update of the Covariance
        # matrix (C) with the rank mu update
        self._parent_pop_size = self._population_size // 2

        # Weights, all set equal for the moment (not sure how they are actually defined)
        # Sum of all positive weights should be 1
        self._W = [math.log((self._population_size + 1) / 2.) - math.log(i) for i in
                   range(1, self._population_size + 1)]

        # Inverse of the Sum of the first parent weights squared (variance effective selection mass)
        self._muEff = np.sum(self._W[:self._parent_pop_size]) ** 2 / np.sum(np.square(self._W[:self._parent_pop_size]))

        # Inverse of the Sum of the last weights squared (variance effective selection mass)
        self._muEffMinus = np.sum(self._W[self._parent_pop_size:]) ** 2 / np.sum(
            np.square(self._W[self._parent_pop_size:]))

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
        self._cmu = min(2 * (self._muEff - 2 + 1 / self._muEff) / ((self._n_parameters + 2) ** 2 + self._muEff)
                        , 1 - self._c1)

        # Damping of the step-size (sigma0) update
        self._dsig = 1 + 2 * max(0., math.sqrt((self._muEff - 1) / (self._n_parameters + 1)) - 1) + self._csig

        # Parameters from the Table 1 of [1]
        alpha_mu = 1 + self._c1 / self._cmu
        alpha_mueff = 1 + 2 * self._muEffMinus / (self._muEff + 2)
        alpha_pos_def = (1 - self._c1 - self._cmu) / (self._n_parameters * self._cmu)

        # Rescaling the weights
        sum_pos = sum([self._W[i] if self._W[i] > 0 else 0 for i in range(self._population_size)])
        sum_neg = sum([self._W[i] if self._W[i] < 0 else 0 for i in range(self._population_size)])

        self._W = [self._W[i] / sum_pos
                   if self._W[i] >= 0
                   else self._W[i] * min(alpha_mu, alpha_mueff, alpha_pos_def) / -sum_neg
                   for i in range(self._population_size)]

        # CMAES always seeds np.random, whether you ask it too or not, so to
        # get consistent debugging output, we should always pass in a seed.
        # Instead of using a fixed number (which would be bad), we can use a
        # randomly generated number: This will ensure pseudo-randomness, but
        # produce consistent results if np.random has been seeded before
        # calling.
        self._seed = 2 ** 31

        # Update optimiser state
        self._running = True

    def name(self):
        """ See :meth:`Optimiser.name()`. """
        return 'Covariance Matrix Adaptation Evolution Strategy (CMA-ES)'

    def running(self):
        """ See :meth:`Optimiser.running()`. """
        return self._running

    def stop(self):
        diag_D = np.diagonal(self._D)
        # We use the condition number defined in the pycma code at
        # https://github.com/CMA-ES/pycma/blob/3abf6900e04d0619f4bfba989dde9e093fa8e1ba/cma/evolution_strategy.py#L2965
        if (np.max(diag_D) / np.min(diag_D)) ** 2 > 1e14:
            return 'Ill-conditionned covariance matrix'
        return False

    def _suggested_population_size(self):
        """ See :meth:`Optimiser._suggested_population_size(). """
        return 4 + int(3 * np.log(self._n_parameters))

    def tell(self, fx):
        """ See :meth:`Optimiser.tell()`. """
        if not self._ready_for_tell:
            raise Exception('ask() not called before tell()')
        self._ready_for_tell = False

        self._counter += 1

        fx[fx == np.inf] = sys.maxsize

        # Get the best xs according to the fx results
        order = np.argsort(fx)
        xs_bests = np.array(self._user_xs[order])
        zs_bests = np.array(self._zs[order])
        ys_bests = np.array(self._ys[order])  # = np.array((xs_bests - self._x0) / self._sigma0)

        # Update the mean
        self._x0 = self._x0 + self._cm * np.sum(np.multiply((xs_bests[:self._parent_pop_size] - self._x0).T,
                                                            self._W[:self._parent_pop_size]).T, 0)

        # Normalizing constants for the evolution path udpate
        norm_cst_sig = math.sqrt(self._csig * (2 - self._csig) * self._muEff)
        norm_cst_c = math.sqrt(self._ccov * (2 - self._ccov) * self._muEff)

        # Get the weighted means of y and z
        zmeans = np.sum(np.multiply(zs_bests[:self._parent_pop_size].T, self._W[:self._parent_pop_size]).T, 0)
        ymeans = np.sum(np.multiply(ys_bests[:self._parent_pop_size].T, self._W[:self._parent_pop_size]).T, 0)

        # Evolution path of sigma (the step size)
        # Note that self._B.dot(zmeans) = self._B.dot(np.linalg.inv(self._D)).dot(self._B.T).dot(ymeans)
        self._psig = (1 - self._csig) * self._psig + norm_cst_sig * self._B.dot(zmeans)

        # In https://github.com/CMA-ES/pycma/blob/3abf6900e04d0619f4bfba989dde9e093fa8e1ba/cma/sigma_adaptation.py#L71
        # They are NOT using exp_size_N0I, but rather a term based on n (number of params)
        exp_size_N0I = (math.sqrt(2) * math.gamma((self._n_parameters + 1) / 2) / math.gamma(self._n_parameters / 2))

        # Heaviside function helps to stall the of pc if norm ||psig|| is too large.
        # This helps to prevent too fast increases in the axes of C when the step size are too small.
        h_sig = 1 if np.linalg.norm(self._psig) / math.sqrt(1 - (1 - self._csig) ** (2 * (self._counter + 1))) < \
                     (1.4 + 2 / (self._n_parameters + 1)) * exp_size_N0I else 0

        delta_sig = (1 - h_sig) * self._ccov * (2 - self._ccov)

        # Evolution path for the rank-1 update
        self._pc = (1 - self._ccov) * self._pc + h_sig * norm_cst_c * ymeans

        # Weight changes taken from the tutorial (no explanation is given for the change)
        # these weights are used for the rank mu update only
        # They allow to keep positive definiteness according to
        # https://github.com/CMA-ES/pycma/blob/3abf6900e04d0619f4bfba989dde9e093fa8e1ba/cma/purecma.py#L419
        temp_weights = [self._W[i] if self._W[i] >= 0
                        else self._W[i] * self._n_parameters / (np.linalg.norm(self._B * zs_bests[i]) ** 2)
                        for i in range(self._population_size)]

        # Update the Covariance matrix:
        # First carry on some of the previous value
        # Add the rank 1 update using the Evolution path
        # Add the rank-mu update
        rank1 = self._c1 * np.outer(self._pc, self._pc)
        rankmu = self._cmu * np.sum(np.multiply(np.array([np.outer(y, y) for y in ys_bests]).T,
                                                temp_weights).T, 0)

        self._C = (1 + delta_sig * self._c1 - self._c1 - self._cmu * sum(self._W)) * self._C + rank1 + rankmu

        # Update of the step size
        # Here we are simply looking at the ratio of the length of the evolution path
        # vs its expected lenght ( E|Gaussian(0,I)|)
        # We use the difference of the ratio with 1 and scale using the learning rate and the damping parameters.
        # At https://github.com/CMA-ES/pycma/blob/3abf6900e04d0619f4bfba989dde9e093fa8e1ba/cma/evolution_strategy.py#L1622
        # They use exp_size_N0I = N**0.5 * (1 - 1. / (4.*N) + 1. / (21.*N**2))
        exp_size_N0I = (math.sqrt(2) * math.gamma((self._n_parameters + 1) / 2) / math.gamma(self._n_parameters / 2))

        self._sigma0 = self._sigma0 * math.exp(self._csig / self._dsig *
                                               (np.linalg.norm(self._psig) / exp_size_N0I - 1))

        # Update B and D
        self._C = np.triu(self._C) + np.triu(self._C, 1).T
        [eigenvals, self._B] = np.linalg.eigh(self._C)

        self._D = np.sqrt(np.diag(eigenvals))

        if self.stop():
            return False

        if self._fbest > fx[order[0]]:
            self._fbest = fx[order[0]]
            self._xbest = xs_bests[0]

    def print_all_info(self):
        print("parents weights ", self._W[:self._parent_pop_size])
        print("other weights", self._W[self._parent_pop_size:])
        print("c1", self._c1)
        print("cmu", self._cmu)
        print("ccov", self._ccov)
        print("csig", self._csig)
        print("Mean", self._x0)
        print("Covariance matrix", self._C)
        print("Step size", self._sigma0)
        print("B or EIGENVECTORS", self._B)
        print("D", self._D)

    def get_cov(self):
        return np.copy(self._C)

    def get_B(self):
        return np.copy(self._B)

    def get_D(self):
        return np.copy(self._D)

    def get_mean(self):
        return np.copy(self._x0)

    def get_sigma(self):
        return np.copy(self._sigma0)

    def set_cov(self, cov):
        self._C = cov

    def set_mean(self, mean):
        self._x0 = mean

    def set_sigma(self, sigma):
        self._sigma0 = sigma

    def xbest(self):
        """ See :meth:`Optimiser.xbest()`. """
        if self._running:
            return np.array(self._xbest, copy=True)
        return np.array([float('inf')] * self._n_parameters)