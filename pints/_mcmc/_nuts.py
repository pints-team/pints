#
# NUTS MCMC method
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
import random


class NUTSMCMC(pints.SingleChainMCMC):
    r"""
    Implements No U-Turn Sampler (NUTS) as described in Algorithm 3 in [1]_.
    Like Hamiltonian Monte Carlo, NUTS imagines a particle moving over
    negative log-posterior (NLP) space to generate proposals. Naturally, the
    particle tends to move to locations of low NLP -- meaning high posterior
    density. Unlike HMC, NUTS allows the number of steps taken through
    parameter space to depend on position, allowing local adaptation.

    The algorithm consists of the following steps in each iteration (t)::

        r_0 ~ N(0, I)
        u ~ uniform(0, exp(L(theta^t-1) - 0.5 r_0.r_0))
        initialise:
            theta_minus = theta^t-1
            theta_plus = theta^t-1
            theta_t = theta^t-1
            r_minus = r_0
            r_plus = r_0
            j = 0
            n = 1
            s = 1
        while s=1 do:
            v_j ~ discrete_uniform({-1, 1})
            if v_j = -1:
                theta_minus, r_minus, _, _, theta_primed, n_primed, s_primed =
                    BuildTree(theta_minus, r_minus, u, v_j, j, epsilon)
            else:
                _, _, theta_plus, r_plus, theta_primed, n_primed, s_primed =
                    BuildTree(theta_plus, r_plus, u, v_j, j, epsilon)
            endif

            if s_primed = 1:
                u_1 ~ uniform(0, 1)
                if n_primed / n > u_1:
                    theta_t = theta_primed
                endif
            endif
            n = n + n_primed
            s = s_primed * 1((theta_plus - theta_minus).r_minus >= 0) *
                1((theta_plus - theta_minus).r_plus >= 0)
            j = j + 1
        endwhile

    where epsilon is the step size chosen by user and 1(x) is the indicator
    function: equal 1 if x is true; 0 otherwise. The BuildTree function is
    given by::

        function BuildTree(theta, r, u, v, j, epsilon):
            if j = 0:
                theta_primed, r_primed = Leapfrog(theta, r, v epsilon)
                n_primed = 1(u <= exp(L(theta_primed) - 0.5 r_primed.r_primed))
                s_primed = 1((L(theta_primed) - 0.5 r_primed.r_primed) >
                             log u - Delta_max)
                theta_minus = theta_primed
                r_minus = r_primed
                theta_plus = theta_primed
                r_plus = r_primed
            else:
                theta_minus, r_minus, theta_plus, r_plus, theta_primed_1,
                    n_primed_1, s_primed_1 = BuildTree(theta, r, u, v,
                                                       j - 1, epsilon)
                if s_primed_1 = 1:
                    if v = -1:
                        theta_minus, r_minus, _, _, theta_primed_1, n_primed_1,
                            s_primed_1 = BuildTree(theta_minus, r_minus,
                                                   u, v, j - 1, epsilon)
                    else:
                        _, _, theta_plus, r_plus, theta_primed_1, n_primed_1,
                            s_primed_1 = BuildTree(theta_plus, r_plus,
                                                   u, v, j - 1, epsilon)
                    endif
                    u_1 ~ uniform(0, 1)
                    if n_primed_1 / (n_primed + n_primed_1) > u_1:
                        theta_primed = theta_primed_1
                    endif
                    s_primed = s_primed_1 *
                               1((theta_plus - theta_minus).r_minus >= 0) *
                               1((theta_plus - theta_minus).r_plus >= 0)
                    n_primed = n_primed + n_primed_1
                endif
            endif
            return theta_minus, r_minus, theta_plus, r_plus, theta_primed,
                   n_primed, s_primed

    where Delta_max=1000 and the below function performs sympletic stepping::

        function Leapfrog(theta, r, epsilon)
            r_tilde = r + (epsilon / 2) * dL(theta)/dtheta
            theta_tilde = theta + epsilon * r_tilde
            r_tilde = r_tilde + (epsilon / 2) * dL(theta_tilde)/dtheta
        return theta_tilde, r_tilde

    Extends :class:`SingleChainMCMC`.

    References
    ----------
    .. [1] "The No-U-Turn Sampler: Adaptively Setting Path Lengths in
           Hamiltonian Monte Carlo", Journal of Machine Learning Research,
           Matthew D Hoffman and Andrew Gelman, 2014.
    """
    def __init__(self, x0, sigma0=None):
        super(NUTSMCMC, self).__init__(x0, sigma0)

        # Set initial state
        self._running = False
        self._ready_for_tell = False

        # Current point in the Markov chain
        self._current = None            # Aka current_q in the chapter
        self._current_U = None     # Aka U(current_q) = -log_pdf
        self._current_gradient = None
        self._current_momentum = None   # Aka current_p
        self._Delta_max = 1000
        self._depth = None
        self._first_leapfrog = True
        self._max_tree_depth = 10
        self._log_pdf = None
        self._n = None
        self._n_primed = None
        self._r_minus = None
        self._r_plus = None
        self._s = None
        self._s_primed = None
        self._theta_minus = None
        self._theta_plus = None
        self._theta_tilde = None
        self._v = None

        # Current point in the leapfrog iterations
        self._momentum = None       # Aka p in the chapter
        self._position = None       # Aka q in the chapter
        self._gradient = None       # Aka grad_U(q) in the chapter

        # Iterations, acceptance monitoring, and leapfrog iterations
        self._mcmc_iteration = 0
        self._mcmc_acceptance = 0

        # Default integration step size for leapfrog algorithm
        self._epsilon = 0.1
        self._step_size = None
        self.set_leapfrog_step_size(np.diag(self._sigma0))

        # Divergence checking
        # Create a vector of divergent iterations
        self._divergent = np.asarray([], dtype='int')

        # Default threshold for Hamiltonian divergences
        # (currently set to match Stan)
        self._hamiltonian_threshold = 10**3

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """
        # Check ask/tell pattern
        if self._ready_for_tell:
            raise RuntimeError('Ask() called when expecting call to tell().')

        # Initialise on first call
        if not self._running:
            self._running = True

        # Very first iteration
        if self._current is None:

            # Ask for the pdf and gradient of x0
            self._ready_for_tell = True
            return np.array(self._x0, copy=True)

        # start NUTS iteration
        if self._within_iteration is False:
            self._within_iteration = True
            self._current_momentum = np.random.multivariate_normal(
                np.zeros(self._n_parameters), np.eye(self._n_parameters))
            r_sq = np.linalg.norm(self._current_momentum)**2
            log_u = self._log_pdf - (0.5 * r_sq -
                                     np.random.exponential(1))
            self._theta_minus = np.copy(self._current)
            self._theta_plus = np.copy(self._current)
            self._r_minus = np.copy(self._current_momentum)
            self._r_plus = np.copy(self._current_momentum)
            self._position = np.copy(self._current)
            self._depth = 0
            self._n = 1
            self._s = 1

        # terminate iteration if max tree depth reached
        if self._depth < self._max_tree_depth:
            # on first leapfrog move one Euler step and return theta
            if self._first_leapfrog:
                if self._s == 1:
                    self._v = random.sample([-1, 1], 1)[0]
                    if self._v == -1:
                        theta, _, _, _, _, _, _ = (
                            self.build_tree(self._theta_minus, self._r_minus,
                                            log_u, self._v, self._depth,
                                            self._scaled_epsilon))
                    else:
                        theta, _, _, _, _, _, _ = (
                            self.build_tree(self._theta_plus, self._r_plus,
                                            log_u, self._v, self._depth,
                                            self._scaled_epsilon))
                return theta
            else:
                if self._v == -1:
                    temp_list = self._build_tree(self._theta_minus,
                                                 self._r_minus, log_u, self._v,
                                                 self._depth,
                                                 self._scaled_epsilon)
                    self._theta_minus = temp_list[0]
                    self._r_minus = temp_list[1]
                else:
                    temp_list = self._build_tree(self._theta_plus,
                                                 self._r_plus, log_u,
                                                 self._v, self._depth,
                                                 self._scaled_epsilon)
                    self._theta_plus = temp_list[2]
                    self._r_plus = temp_list[3]

                self._theta_primed = temp_list[4]
                self._n_primed = temp_list[5]
                self._s_primed = temp_list[6]
                if self._s_primed == 1:
                    u1 = np.random.uniform(0, 1)
                    n_ratio = self._n_primed / self._n
                    if n_ratio > u1:
                        self._position = np.copy(self._theta_primed)
                self._n += self._n_primed
                self._s = self._calculate_s(self._s_primed, self._theta_plus,
                                            self._theta_minus, self._r_plus,
                                            self._r_minus)
                self._depth += 1
        self._within_iteration = False
        return self._position

    def build_tree(self, theta, r, log_u, v, j, epsilon):
        """
        Builds tree containing leaves of position-momenta as defined in
        Algorithm 3 in [1]_.
        """
        if j == 0:
            theta_primed, r_primed = self.leapfrog(theta, r, v * epsilon)
            # if in midst of leapfrog then return just theta to evaulate and
            # pad with Nones to be of same shape as other returns of function
            if not self._first_leapfrog:
                return theta_primed, None, None, None, None, None, None
            l_rhs = self._log_pdf - 0.5 * np.linalg.norm(r_primed)**2
            n_primed = int(log_u <= l_rhs)
            s_primed = int(log_u - self._Delta_max <= l_rhs)
            theta_minus = theta_primed
            r_minus = r_primed
            theta_plus = theta_primed
            r_plus = r_primed
        else:
            (theta_minus, r_minus, theta_plus, r_plus, theta_primed_1,
             n_primed_1, s_primed_1) = self._build_tree(theta, r, log_u, v,
                                                        j - 1, epsilon)
            if s_primed_1 == 1:
                if v == -1:
                    (theta_minus, r_minus, _, _, theta_primed_1, n_primed_1,
                     s_primed_1) = self._build_tree(theta_minus, r_minus,
                                                    log_u, v, j - 1, epsilon)
                else:
                    _, _, theta_plus, r_plus, theta_primed_1, n_primed_1,
                    s_primed_1 = self._build_tree(theta_plus, r_plus,
                                                  log_u, v, j - 1, epsilon)
                u_1 = np.random.uniform(0, 1)
                if n_primed_1 / (n_primed + n_primed_1) > u_1:
                    theta_primed = theta_primed_1
                s_primed = self._calculate_s(s_primed_1, theta_plus,
                                             theta_minus, r_plus, r_minus)
                n_primed = n_primed + n_primed_1
        return theta_minus, r_minus, theta_plus, r_plus, theta_primed,
        n_primed, s_primed

    def _calculate_s(self, s, theta_plus, theta_minus, r_plus, r_minus):
        """
        Calculates s variable as in Algorithm 3 in [1]_.
        """
        theta_diff = theta_plus - theta_minus
        prod_minus = np.dot(theta_diff, r_minus)
        prod_plus = np.dot(theta_diff, r_plus)
        return s * int(prod_plus >= 0) * int(prod_minus >= 0)

    def current_log_pdf(self):
        """ See :meth:`SingleChainMCMC.current_log_pdf()`. """
        return -self._current_U

    def divergent_iterations(self):
        """
        Returns the iteration number of any divergent iterations
        """
        return self._divergent

    def epsilon(self):
        """
        Returns epsilon used in leapfrog algorithm
        """
        return self._epsilon

    def hamiltonian_threshold(self):
        """
        Returns threshold difference in Hamiltonian value from one iteration to
        next which determines whether an iteration is divergent.
        """
        return self._hamiltonian_threshold

    def leapfrog(self, theta, r, epsilon):
        """
        Performs leapfrog steps as defined in Algorithm 1 in [1]_.
        """
        if self._first_leapfrog:
            self._momentum = r + (epsilon / 2) * self._gradient
            self._theta_tilde = theta + epsilon * self._momentum
            self._first_leapfrog = False
        else:
            self._momentum += (epsilon / 2) * self._gradient
            self._first_leapfrog = True
        return self._theta_tilde, self._momentum

    def leapfrog_step_size(self):
        """
        Returns the step size for the leapfrog algorithm.
        """
        return self._step_size

    def _log_init(self, logger):
        """ See :meth:`Loggable._log_init()`. """
        logger.add_float('Accept.')

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        logger.log(self._mcmc_acceptance)

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 2

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Hamiltonian Monte Carlo'

    def needs_sensitivities(self):
        """ See :meth:`pints.MCMCSampler.needs_sensitivities()`. """
        return True

    def scaled_epsilon(self):
        """
        Returns scaled epsilon used in leapfrog algorithm
        """
        return self._scaled_epsilon

    def _set_scaled_epsilon(self):
        """
        Rescales epsilon along the dimensions of step_size
        """
        self._scaled_epsilon = np.zeros(self._n_parameters)
        for i in range(self._n_parameters):
            self._scaled_epsilon[i] = self._epsilon * self._step_size[i]

    def set_epsilon(self, epsilon):
        """
        Sets epsilon for the leapfrog algorithm
        """
        epsilon = float(epsilon)
        if epsilon <= 0:
            raise ValueError('epsilon must be positive for leapfrog algorithm')
        self._epsilon = epsilon
        self._set_scaled_epsilon()

    def set_hamiltonian_threshold(self, hamiltonian_threshold):
        """
        Sets threshold difference in Hamiltonian value from one iteration to
        next which determines whether an iteration is divergent.
        """
        if hamiltonian_threshold < 0:
            raise ValueError('Threshold for divergent iterations must be ' +
                             'non-negative.')
        self._hamiltonian_threshold = hamiltonian_threshold

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[leapfrog_step_size]``.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_leapfrog_step_size(x[0])

    def set_leapfrog_step_size(self, step_size):
        """
        Sets the step size for the leapfrog algorithm.
        """
        a = np.atleast_1d(step_size)
        if len(a[a < 0]) > 0:
            raise ValueError(
                'Step size for leapfrog algorithm must' +
                'be greater than zero.'
            )
        if len(a) == 1:
            step_size = np.repeat(step_size, self._n_parameters)
        elif not len(step_size) == self._n_parameters:
            raise ValueError(
                'Step size should either be of length 1 or equal to the' +
                'number of parameters'
            )
        self._step_size = step_size
        self._set_scaled_epsilon()

    def set_max_tree_deptj(self, max_tree_depth):
        """
        Sets the maximum tree depth, above which the NUTS iteration terminates
        early.
        """
        max_tree_depth = int(max_tree_depth)
        if max_tree_depth <= 0:
            raise ValueError('Max tree depth must be positive')
        self._max_tree_depth = max_tree_depth

    def tell(self, reply):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """
        if not self._ready_for_tell:
            raise RuntimeError('Tell called before proposal was set.')
        self._ready_for_tell = False

        # Unpack reply
        neg_U, gradient = reply

        # Check reply, copy gradient
        neg_U = float(neg_U)
        gradient = pints.vector(gradient)
        assert(gradient.shape == (self._n_parameters, ))

        # Energy = -log_pdf, so flip sign
        U = -neg_U
        self._log_pdf = neg_U

        # Very first call
        if self._current is None:

            # Check first point is somewhere sensible
            if not np.isfinite(U):
                raise ValueError(
                    'Initial point for MCMC must have finite logpdf.')

            # Set current sample, energy, and gradient
            self._current = self._x0
            self._current_U = U
            self._current
            self._current_gradient = gradient

            # Increase iteration count
            self._mcmc_iteration += 1

            # Mark current as read-only, so it can be safely returned
            self._current.setflags(write=False)

            # Return first point in chain
            return self._current

        # Set gradient of current leapfrog position
        self._gradient = gradient

        # Not the last iteration? Then return None
        if self._within_iteration:
            return None

        # Before starting accept/reject procedure, check if the leapfrog
        # procedure has led to a finite momentum and logpdf. If not, reject.
        accept = 0
        if np.isfinite(U) and np.all(np.isfinite(self._momentum)):

            # Evaluate potential and kinetic energies at start and end of
            # leapfrog trajectory
            current_K = np.sum(self._current_momentum**2 / 2)
            proposed_U = U
            proposed_K = np.sum(self._momentum**2 / 2)

            # Check for divergent iterations by testing whether the
            # Hamiltonian difference is above a threshold
            div = proposed_U + proposed_K - (self._current_U + current_K)
            if np.abs(div) > self._hamiltonian_threshold:  # pragma: no cover
                self._divergent = np.append(
                    self._divergent, self._mcmc_iteration)
                self._momentum = self._position = self._gradient = None

                # Update MCMC iteration count
                self._mcmc_iteration += 1

                # Update acceptance rate (only used for output!)
                self._mcmc_acceptance = (
                    (self._mcmc_iteration * self._mcmc_acceptance + accept) /
                    (self._mcmc_iteration + 1))
                self._current.setflags(write=False)
                return self._current

            # always accepts here
            else:
                # acceptance handled in ask
                accept = np.array_equal(self._position, self._current)
                self._current = np.copy(self._position)
                self._current_U = U
                self._current_gradient = gradient

                # Mark current as read-only, so it can be safely returned
                self._current.setflags(write=False)

        # Reset leapfrog mechanism
        self._momentum = self._position = self._gradient = None

        # Update MCMC iteration count
        self._mcmc_iteration += 1

        # Update acceptance rate (only used for output!)
        self._mcmc_acceptance = (
            (self._mcmc_iteration * self._mcmc_acceptance + accept) /
            (self._mcmc_iteration + 1))

        # Return current position as next sample in the chain
        return self._current
