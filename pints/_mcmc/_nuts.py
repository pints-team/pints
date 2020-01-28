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
import copy


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
                #print('accepting with prob',p)
                #print('root n_dash = {}, n = {}'.format(n_dash,self.n))
            else:
                p = n_dash / (self.n + n_dash)

            if p > 0.0 and np.random.uniform() < p:
                self.theta = theta_dash
                self.L = L_dash

        if root:
            self.alpha = alpha_dash
            self.n_alpha = n_alpha_dash
        else:
            #print('adding {} to {}'.format(alpha_dash,self.alpha))
            self.alpha += alpha_dash
            self.n_alpha += n_alpha_dash

        self.n += n_dash
        self.s *= s_dash
        self.s *= int((self.theta_plus - self.theta_minus).dot(self.r_minus) >= 0)
        self.s *= int((self.theta_plus - self.theta_minus).dot(self.r_plus) >= 0)
        #print('tests {} {}'.format((self.theta_plus - self.theta_minus).dot(self.r_minus), (self.theta_plus - self.theta_minus).dot(self.r_plus)))

        #print('updating state, new state is theta_minus = {}, theta_plus = {} theta = {}, n = {}, s = {}'.format(self.theta_minus,self.theta_plus,self.theta,self.n,self.s))

@asyncio.coroutine
def leapfrog(theta, r, epsilon, step_size):
    #print('leapfrog from ({}, {})'.format(theta,r))
    #print('theta = ',theta)
    L, grad_L = (yield theta)

    #print('grad_L = ',grad_L)
    r_new = r + 0.5*epsilon*step_size*grad_L
    theta_new = theta + epsilon*step_size*r_new
    L_new, grad_L_new = (yield theta_new)
    r_new += 0.5*epsilon*step_size*grad_L_new
    #print('leapfrog to ({}, {})'.format(theta_new,r_new))
    #print('grad_L_new = {}, step_size = {}'.format(grad_L_new, step_size))
    return L_new, theta_new, r_new


@asyncio.coroutine
def build_tree(state, log_u, v, j, epsilon, hamiltonian0, step_size):
    if j == 0:
        # Base case - take one leapfrog in the direction v
        if v == -1:
            theta = state.theta_minus
            r = state.r_minus
        else:
            theta = state.theta_plus
            r = state.r_plus

        L_dash, theta_dash, r_dash = yield from leapfrog(theta, r, v*epsilon, step_size)
        if np.isnan(r_dash).any():
            r_dash = np.zeros_like(theta)
        hamiltonian_dash = L_dash - 0.5*r_dash.dot(r_dash)
        #print('theta_dash = {}, log_u = {}, hamiltonian = {}'.format(theta_dash,log_u,hamiltonian_dash))
        n_dash = int(log_u <= hamiltonian_dash)
        #n_dash = 1
        #print('n_dash',n_dash)
        comparison = hamiltonian_dash - hamiltonian0
        Delta_max = 1000
        s_dash = int(log_u < Delta_max + hamiltonian_dash)
        #print('build_tree base case, s_dash = {}'.format(s_dash))
        #print('comparison',comparison)
        #print('epsilon',epsilon)
        #print('step_size',step_size)
        alpha_dash = min(1.0, np.exp(comparison))
        #print('alpha_dash',alpha_dash)
        n_alpha_dash = 1
        #print('s_dash = {}, u = {}, hamiltonian = {}'.format(s_dash, u, hamiltonian_dash))
        return nuts_state(
                theta_dash, theta_dash,
                r_dash, r_dash,
                theta_dash, L_dash, n_dash, s_dash,
                alpha_dash, n_alpha_dash
                )

    else:
        # Recursion - implicitly build the left and right subtrees
        state_dash = yield from build_tree(state, log_u, v, j-1, epsilon, hamiltonian0, step_size)

        if state_dash.s == 1:
            state_double_dash = yield from build_tree(state_dash, log_u, v, j-1, epsilon, hamiltonian0,
                    step_size)
            state_dash.update(state_double_dash, direction=v, root=False)

        return copy.copy(state_dash)


@asyncio.coroutine
def find_reasonable_epsilon(theta, L, step_size):
    epsilon = 1.0
    r = np.random.normal(size=len(theta))
    hamiltonian = L - 0.5*r.dot(r)

    L_dash, theta_dash, r_dash = yield from leapfrog(theta, r, epsilon, step_size)

    # r_dash could be nan in unfeasable regions
    if np.isnan(r_dash).any():
        r_dash[:] = 0.0
    hamiltonian_dash = L_dash - 0.5*r_dash.dot(r_dash)
    comparison = hamiltonian_dash - hamiltonian
    alpha = 2 * int(comparison > np.log(0.5)) - 1
    # sometimes the np.exp(-inf) gives a nan, sometimes 0, not sure why???
    while comparison * alpha > np.log(2) * (-alpha):
        epsilon = 2**alpha * epsilon
        L_dash, theta_dash, r_dash = yield from leapfrog(theta, r, epsilon, step_size)
        if np.isnan(r_dash).any():
            r_dash[:] = 0.0
        hamiltonian_dash = L_dash - 0.5*r_dash.dot(r_dash)
        comparison = hamiltonian_dash - hamiltonian
    print('reasonable epsilon',epsilon)
    return epsilon


@asyncio.coroutine
def nuts_sampler(x0, delta, M_adapt, step_size):
    theta = x0
    L, grad_L = (yield theta)
    epsilon = yield from find_reasonable_epsilon(theta, L, step_size)
    #epsilon = 1.0
    mu = np.log(10*epsilon)
    log_epsilon_bar = np.log(1)
    H_bar = 0
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    m = 1

    while True:
        r0 = np.random.normal(size=len(theta))
        hamiltonian0 = L - 0.5*r0.dot(r0)
        log_u = np.log(np.random.uniform(0, 1)) + hamiltonian0
        #u = np.random.uniform(0, np.exp(hamiltonian0))
        #log_u = np.log(u)
        #print('generated log_u = {}, hamiltonian0 = {}'.format(log_u, hamiltonian0))
        state = nuts_state(theta, theta, r0, r0, theta, L, 1, 1, None, None)
        j = 0
        while j < 10 and state.s == 1:
            # pick a direction
            if np.random.randint(0,2):
                vj = 1
            else:
                vj = -1

            # recursivly build up tree in that direction
            state_dash = yield from build_tree(state, log_u, vj, j, epsilon, hamiltonian0,
                    step_size)
            state.update(state_dash, direction=vj, root=True)

            #print('vj={}, j = {} and n_dash = {}, s_dash = {}'.format(vj, j,state_dash.n,state_dash.s))
            j += 1

        # adaption
        if m < M_adapt:
            H_bar = (1 - 1.0/(m+t0)) * H_bar + 1.0/(m+t0) * (delta - state.alpha/state.n_alpha)
            log_epsilon = mu - (np.sqrt(m)/gamma) * H_bar
            log_epsilon_bar = m**(-kappa) * log_epsilon + (1 - m**(-kappa)) * log_epsilon_bar
            epsilon = np.exp(log_epsilon)
        elif m == M_adapt:
            epsilon = np.exp(log_epsilon_bar)
        #print(epsilon)

        # update current position
        theta = state.theta
        L = state.L
        hamiltonian_dash = L - 0.5*r0.dot(r0)
        #print('j = {}'.format(j))
        #print('nuts accept prob = ',state.alpha/state.n_alpha)
        #print('nuts alpha = ',state.alpha)
        #print('nuts n_alpha = ',state.n_alpha)
        #print('my accept prob = ',min(1.0,np.exp(hamiltonian_dash-hamiltonian0)))
        #print('hamiltonian0 = {}, hamiltonian_dash = {}'.format(hamiltonian0,hamiltonian_dash))

        # return current position, log pdf, and average acceptance probability to sampler
        #print('epsilon',step_size)
        #print('n_alpha',state.n_alpha)
        yield (theta, L, state.alpha/state.n_alpha)

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
        self._M_adapt = 100
        self._delta = 0.6
        self._step_size = None
        self.set_leapfrog_step_size(np.diag(self._sigma0))
        #self.set_leapfrog_step_size(1.0)

        # coroutine nuts sampler
        self._nuts = None

        # number of mcmc iterations
        self._mcmc_iteration = 0

        # averaged acceptance probability
        self._mcmc_acceptance = 0

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
            self._nuts = nuts_sampler(self._x0, self._delta, self._M_adapt,
                    self._step_size)
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
            # extract next point in chain, its logpdf and acceptance
            # probability
            self._current = self._next[0]
            self._current_logpdf = self._next[1]
            self._current_acceptance = self._next[2]

            # Increase iteration count
            self._mcmc_iteration += 1

            if self._mcmc_iteration <= self._M_adapt:
                # if still adapting report the instantaneous acceptance
                # probability
                self._mcmc_acceptance = self._current_acceptance
            else:
                # after adaption report the averaged acceptance probability
                post_adapt_iterations = self._mcmc_iteration - self._M_adapt
                self._mcmc_acceptance = (
                    (post_adapt_iterations * self._mcmc_acceptance + self._current_acceptance) /
                    (post_adapt_iterations + 1)
                    )

            # request next point to evaluate
            self._next = next(self._nuts)

            # Return current position as next sample in the chain
            return self._current
        else:
            # Return None to indicate there is no new sample for the chain
            return None

    def _log_init(self, logger):
        """ See :meth:`Loggable._log_init()`. """
        logger.add_float('Accept.')

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        logger.log(self._mcmc_acceptance)

    def current_log_pdf(self):
        """ See :meth:`SingleChainMCMC.current_log_pdf()`. """
        return self._current_logpdf

    def delta(self):
        """
        Returns delta used in leapfrog algorithm
        """
        return self._delta

    def leapfrog_step_size(self):
        """
        Returns the step size for the leapfrog algorithm.
        """
        return self._step_size

    def number_adaption_steps(self):
        """
        Returns number of adaption steps used in the NUTS algorithm.
        """
        return self._M_adapt

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 2

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'No-U-Turn MCMC'

    def needs_sensitivities(self):
        """ See :meth:`pints.MCMCSampler.needs_sensitivities()`. """
        return True

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
        print('set step size = {}'.format(step_size))
        self._step_size = step_size

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
        The hyper-parameter vector is ``[delta, number_adaption_steps, leapfrog_step_size]``.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_delta(x[0])
        self.set_number_adaption_steps(x[1])
        self.set_leapfrog_step_size(x[2])



