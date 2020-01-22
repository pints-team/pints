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
import pints
import numpy as np


class nuts_state:
    def __init__(self, theta_minus, theta_plus, r_minus, r_plus, theta, n, s, alpha,
            n_alpha):
        self.theta_minus = theta_minus
        self.theta_plus = theta_plus
        self.r_minus = r_minus
        self.r_plus = r_plus
        self.n = n
        self.s = s
        self.theta = theta
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
        n_dash = other_state.n
        s_dash = other_state.s
        alpha_dash = other_state.alpha
        n_alpha_dash = other_state.n_alpha

        if root:
            p = int(s_dash == 1)*np.min(1, n_dash / n)
            self.alpha = alpha_dash
            self.n_alpha = n_alpha_dash
        else:
            p = n_dash / (self.n + n_dash)
            self.alpha += alpha_dash
            self.n_alpha += n_alpha_dash

        if p > 0.0 and np.random.uniform() < p:
            self.theta = theta_dash

        self.n += n_dash
        self.s *= s_dash
        self.s *= int((self.theta_plus - self.theta_minus)*self.r_minus >= 0)
        self.s *= int((self.theta_plus - self.theta_minus)*self.r_plus >= 0)

@asyncio.coroutine
def leapfrog(theta, r, epsilon):
    grad_L = (yield ('grad_L', theta))
    r_new = r + 0.5*epsilon*grad_L
    theta_new = theta + epsilon*r_new
    grad_L_new = (yield ('grad_L', theta_new))
    r_new += 0.5*epsilon*grad_L_new
    return theta_new, r_new

@asyncio.coroutine
def build_tree(state, u, v, j, epsilon, L_minus_r_dot_r0):
    if j == 0:
        # Base case - take one leapfrog in the direction v
        if v == -1:
            theta = state.theta_minus
            r = state.r_minus
        else:
            theta = state.theta_plus

        r = state.r_plus
        theta_dash, r_dash = leapfrog(theta, r, v*epsilon)
        L = (yield ('L', theta_dash))
        L_minus_r_dot_r = L - 0.5*r.dot(r)
        n_dash = int(u <= np.exp(L_minus_r_dot_r))
        comparison = L_minus_r_dot_r - L_minus_r_dot_r0
        s_dash = int(u < np.exp(Delta_max + comparison))
        alpha_dash = min(1, np.exp(comparison))
        n_alpha = 1.0
        return nuts_state(
                theta_dash, r_dash,
                theta_dash, r_dash,
                theta_dash, n_dash, s_dash,
                alpha, n_alpha
                )
    else:
        # Recursion - implicitly build the left and right subtrees
        state = build_tree(state, u, v, j-1, eta, L_minus_r_dot_r0)

        if state.s == 1:
            state.update(
                    build_tree(state, u, v, j-1, eta, L_minus_r_dot_r0),
                    direction=vj, root=False
                    )

        return state


@asyncio.coroutine
def nuts_sampler(x0, delta, M_adapt):
    n = len(x0)
    I = np.identity(n)
    zero = np.zeros(n)
    theta = x0
    epsilon = find_reasonable_epsilon(theta)
    mu = np.log(10*epsilon)
    log_epsilon_bar = np.log(1)
    H_bar = 0
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    m = 0
    while True:
        r0 = np.random.normal(zero, I)
        L = (yield ('L', theta))
        L_minus_r_dot_r0 = L - 0.5*r0.dot(r0)
        u = np.random.uniform(0, np.exp(L_minus_r_dot_r0))
        state = nuts_state(theta, theta, r0, r0, theta, 1, 1)
        j = 0
        while state.s == 1:
            # pick a direction
            if np.random.randint(0,2):
                vj = 1
            else:
                vj = -1

            # recursivly build up tree in that direction
            state.update(
                build_tree(state, u, vj, j, epsilon, L_minus_r_dot_r0),
                direction=vj, root=True
                )

        # adaption
        if m < M_adapt:
            H_bar = (1 - 1.0/(m+t0)) * (delta - state.alpha / state.n_alpha)
            log_epsilon = mu - (np.sqrt(m)/gamma) * H_bar
            log_epsilon_bar = m**(-kappa) * log_epsilon + (1 - m**(-kappa)) * log_epsilon_bar
        else if m == M_adapt:
            epsilon = np.exp(log_epsilon_bar)

        # next step
        m += 1





class NoUTurnSamplerMCMC(pints.SingleChainMCMC):
    r"""
    Implements the No-U-Turn Sampler as described in [1]_.

    Extends :class:`SingleChainMCMC`.

    References
    ----------
    .. [1] Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn sampler:
           adaptively setting path lengths in Hamiltonian Monte Carlo.
           Journal of Machine Learning Research, 15(1), 1593-1623.
    """
    def __init__(self, x0, sigma0=None):
        super(NoUTurnSamplerMCMC, self).__init__(x0, sigma0)

        # hyperparameters
        self._M_adapt = 1000
        self._delta = 0.5
        self._nuts = None


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
            self._running = True

        # Ask for the pdf and gradient of the current leapfrog position
        # Using this, the leapfrog step for the momentum is performed in tell()
        self._ready_for_tell = True
        return np.array(next(self._nuts), copy=True)

    def current_log_pdf(self):
        """ See :meth:`SingleChainMCMC.current_log_pdf()`. """
        return -self._current_energy

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
        logger.add_float('Accept.')

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        logger.log(self._mcmc_acceptance)

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 2

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'No-U-Turn Sampler'

    def needs_sensitivities(self):
        """ See :meth:`pints.MCMCSampler.needs_sensitivities()`. """
        return True

    def set_delta(self, epsilon):
        """
        Sets delta for the nuts algorithm
        """
        #TODO

    def set_number_adaption_steps(self, n):
        """
        Sets number of adaptions steps in the nuts algorithm
        """
        #TODO

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[delta, number_adaption_steps]``.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_delta(x[0])
        self.set_number_adaption_steps(x[1])

    def tell(self, reply):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """
        if not self._ready_for_tell:
            raise RuntimeError('Tell called before proposal was set.')
        self._ready_for_tell = False

        #TODO

        # Return None to indicate there is no new sample for the chain
        return None

        # Return current position as next sample in the chain
        return self._current
