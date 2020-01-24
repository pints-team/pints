#
# No-U-Turn Sampler (NUTS) MCMC method
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import asyncio
import pints
import numpy as np


class nuts_state:
    def __init__(self, theta_minus, theta_plus, r_minus, r_plus,
            theta, L, n, s, alpha, n_alpha):
        self.theta_minus = theta_minus
        self.theta_plus = theta_plus
        self.r_minus = r_minus
        self.r_plus = r_plus
        self.n = n
        self.s = s
        self.theta = theta
        self.L = L
        self.alpha = alpha
        self.n_alpha = n_alpha

    def update(self, other_state, direction, root):
        if direction == -1:
            self.theta_minus = other_state.theta_minus
            self.r_minus = other_state.r_minus
        else:
            self.theta_plus = other_state.theta_plus
            self.r_plus = other_state.r_plus

        theta_dash = other_state.theta
        L_dash = other_state.L
        n_dash = other_state.n
        s_dash = other_state.s
        alpha_dash = other_state.alpha
        n_alpha_dash = other_state.n_alpha

        if n_dash > 0:
            if root:
                p = int(s_dash == 1)*min(1, n_dash / self.n)
            else:
                p = n_dash / (self.n + n_dash)

            if p > 0.0 and np.random.uniform() < p:
                self.theta = theta_dash
                self.L = L_dash

        if root:
            self.alpha = alpha_dash
            self.n_alpha = n_alpha_dash
        else:
            self.alpha += alpha_dash
            self.n_alpha += n_alpha_dash

        self.n += n_dash
        self.s *= s_dash
        self.s *= int((self.theta_plus - self.theta_minus).dot(self.r_minus) >= 0)
        self.s *= int((self.theta_plus - self.theta_minus).dot(self.r_plus) >= 0)

        #print('updating state, new state is theta_minus = {}, theta_plus = {} theta = {}, n = {}, s = {}'.format(self.theta_minus,self.theta_plus,self.theta,self.n,self.s))

@asyncio.coroutine
def leapfrog(theta, r, epsilon):
    #print('leapfrog from ({}, {})'.format(theta,r))
    L, grad_L = (yield theta)
    r_new = r + 0.5*epsilon*grad_L
    theta_new = theta + epsilon*r_new
    L_new, grad_L_new = (yield theta_new)
    r_new += 0.5*epsilon*grad_L_new
    #print('leapfrog to ({}, {})'.format(theta_new,r_new))
    return L_new, theta_new, r_new


@asyncio.coroutine
def build_tree(state, u, v, j, epsilon, hamiltonian0):
    if j == 0:
        # Base case - take one leapfrog in the direction v
        if v == -1:
            theta = state.theta_minus
            r = state.r_minus
        else:
            theta = state.theta_plus
            r = state.r_plus

        L_dash, theta_dash, r_dash = yield from leapfrog(theta, r, v*epsilon)
        hamiltonian = L_dash - 0.5*r.dot(r)
        n_dash = int(u <= np.exp(hamiltonian))
        comparison = hamiltonian - hamiltonian0
        Delta_max = 1000
        s_dash = int(np.log(u) < Delta_max + hamiltonian)
        #print('build_tree base case, s_dash = {}'.format(s_dash))
        alpha_dash = min(1, np.exp(comparison))
        n_alpha_dash = 1.0
        return nuts_state(
                theta_dash, r_dash,
                theta_dash, r_dash,
                theta_dash, L_dash, n_dash, s_dash,
                alpha_dash, n_alpha_dash
                )
    else:
        # Recursion - implicitly build the left and right subtrees
        state = yield from build_tree(state, u, v, j-1, epsilon, hamiltonian0)

        if state.s == 1:
            state_dash = yield from build_tree(state, u, v, j-1, epsilon, hamiltonian0)
            state.update(state_dash, direction=v, root=False)

        return state


@asyncio.coroutine
def find_reasonable_epsilon(theta, L):
    epsilon = 1.0
    r = np.random.normal(size=len(theta))
    L_dash, theta_dash, r_dash = yield from leapfrog(theta, r, epsilon)
    p_theta_r = np.exp(L - 0.5*r.dot(r))
    p_theta_r_dash = np.exp(L_dash - 0.5*r_dash.dot(r_dash))
    ratio = p_theta_r_dash/p_theta_r
    alpha = 2 * int(ratio > 0.5) - 1
    while ratio**alpha > 2**(-alpha):
        #print('find_reasonable_epsilon, alpha = {}, epsilon = {} ratio = {}'.format(alpha, epsilon, ratio))
        epsilon = 2**alpha * epsilon
        L_dash, theta_dash, r_dash = yield from leapfrog(theta, r, epsilon)
        p_theta_r_dash = np.exp(L_dash - 0.5*r_dash.dot(r_dash))
        ratio = p_theta_r_dash/p_theta_r
    return epsilon


@asyncio.coroutine
def nuts_sampler(x0, delta, M_adapt):
    theta = x0
    L, grad_L = (yield theta)
    epsilon = yield from find_reasonable_epsilon(theta, L)
    print('reasonable epsilon = {}'.format(epsilon))
    mu = np.log(10*epsilon)
    log_epsilon_bar = np.log(1)
    H_bar = 0
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    m = 1

    while True:
        r0 = np.random.normal(size=len(theta))
        L_minus_r_dot_r0 = L - 0.5*r0.dot(r0)
        u = np.random.uniform(0, np.exp(L_minus_r_dot_r0))
        state = nuts_state(theta, theta, r0, r0, theta, L, 1, 1, None, None)
        j = 0
        while state.s == 1:
            # pick a direction
            if np.random.randint(0,2):
                vj = 1
            else:
                vj = -1

            # recursivly build up tree in that direction
            state_dash = yield from build_tree(state, u, vj, j, epsilon, L_minus_r_dot_r0)
            state.update(state_dash, direction=vj, root=True)

            j += 1

        # adaption
        if m < M_adapt:
            H_bar = (1 - 1.0/(m+t0)) * H_bar + 1.0/(m+t0) * (delta - state.alpha/state.n_alpha)
            log_epsilon = mu - (np.sqrt(m)/gamma) * H_bar
            log_epsilon_bar = m**(-kappa) * log_epsilon + (1 - m**(-kappa)) * log_epsilon_bar
            epsilon = np.exp(log_epsilon)
        elif m == M_adapt:
            epsilon = np.exp(log_epsilon_bar)

        # update current position
        theta = state.theta
        L = state.L
        #print('new state is ({}, {})'.format(theta,L))

        # return current position and log pdf to sampler
        yield (theta, L)

        # next step
        m += 1

class NoUTurnMCMC(pints.SingleChainMCMC):
    r"""

    Implements No U-Turn Sampler (NUTS) with dual averaging, as described in Algorithm 6
    in [1]_. Like Hamiltonian Monte Carlo, NUTS imagines a particle moving over negative
    log-posterior (NLP) space to generate proposals. Naturally, the particle tends to
    move to locations of low NLP -- meaning high posterior density. Unlike HMC, NUTS
    allows the number of steps taken through parameter space to depend on position,
    allowing local adaptation.

    Extends :class:`SingleChainMCMC`.

    References
    ----------
    .. [1] Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn sampler:
           adaptively setting path lengths in Hamiltonian Monte Carlo.
           Journal of Machine Learning Research, 15(1), 1593-1623.
    """
    def __init__(self, x0, sigma0=None):
        super(NoUTurnMCMC, self).__init__(x0, sigma0)

        # hyperparameters
        self._M_adapt = 1000
        self._delta = 0.5
        self._nuts = None

        # current point in chain
        self._current = self._x0

        # current logpdf (last logpdf returned by tell)
        self._current_logpdf = None

        # next point to ask user to evaluate
        self._next = self._current

        # Set initial state
        self._running = False
        self._ready_for_tell = False

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """
        # Check ask/tell pattern
        if self._ready_for_tell:
            raise RuntimeError('Ask() called when expecting call to tell().')

        # Initialise on first call
        if not self._running:
            self._nuts = nuts_sampler(self._x0, self._delta, self._M_adapt)
            # coroutine will ask for self._x0
            self._next = next(self._nuts)
            self._running = True

        self._ready_for_tell = True
        return np.array(self._next, copy=True)

    def tell(self, reply):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """
        if not self._ready_for_tell:
            raise RuntimeError('Tell called before proposal was set.')
        self._ready_for_tell = False

        # send log likelihood and gradient to nuts coroutine, which returns next theta
        # value to evaluate at
        self._next = self._nuts.send(reply)

        # coroutine signals end of current step by sending (theta, L), where
        # theta is the next point in the chain and L is its log-pdf
        if isinstance(self._next, tuple):
            self._current = self._next[0]
            self._current_logpdf = self._next[1]

            # request next point to evaluate
            self._next = next(self._nuts)

            # Return current position as next sample in the chain
            return self._current
        else:
            # Return None to indicate there is no new sample for the chain
            return None


    def current_log_pdf(self):
        """ See :meth:`SingleChainMCMC.current_log_pdf()`. """
        return self._current_logpdf

    def delta(self):
        """
        Returns delta used in leapfrog algorithm
        """
        return self._delta

    def number_adaption_steps(self):
        """
        Returns number of adaption steps used in the NUTS algorithm.
        """
        return self._M_adapt

    def _log_init(self, logger):
        """ See :meth:`Loggable._log_init()`. """
        pass

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        pass

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 2

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'No-U-Turn MCMC'

    def needs_sensitivities(self):
        """ See :meth:`pints.MCMCSampler.needs_sensitivities()`. """
        return True

    def set_delta(self, delta):
        """
        Sets delta for the nuts algorithm
        """
        if self._running:
            raise RuntimeError('cannot set delta while sampler is running')
        if delta < 0 or delta > 1:
            raise ValueError('delta must be in [0, 1]')
        self._delta = delta

    def set_number_adaption_steps(self, n):
        """
        Sets number of adaptions steps in the nuts algorithm
        """
        if self._running:
            raise RuntimeError('cannot set number of adaption steps while sampler is running')
        if n < 0:
            raise ValueError('number of adaption steps must be non-negative')
        self._M_adapt = int(n)

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[delta, number_adaption_steps]``.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_delta(x[0])
        self.set_number_adaption_steps(x[1])



