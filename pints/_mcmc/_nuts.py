#
# No-U-Turn Sampler (NUTS) MCMC method
#
# This file is part of PINTS.
#  Copyright (c) 2017-2020, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import asyncio
import pints
import numpy as np


class NutsState:
    """
    Class to hold information about the current state of the NUTS hamiltonian
    integration path.

    NUTS builds up the integration path implicitly via recursion up a binary
    tree, this class handles combining states from different subtrees (see
    `update`).

    NUTS integrates both backwards ("minus") and forwards ("plus"), so this
    state must keep track of both end points of the integration path.

    Attributes
    ----------

    theta_minus: ndarray
        parameter value at the backwards end of the integration path

    theta_plus: ndarray
        parameter value at the forwards end of the integration path

    r_minus: ndarray
        momentum value at the backwards end of the integration path

    r_plus: ndarray
        momentum value at the forwards end of the integration path

    L_minus: float
        logpdf value at the backwards end of the integration path

    L_plus: float
        logpdf value at the forwards end of the integration path

    grad_L_minus: float
        gradient of logpdf at the backwards end of the integration path

    grad_L_plus: float
        gradient of logpdf at the forwards end of the integration path

    n: int or float
        number of accepted points in the path. If the sampler is using
        multinomial_sampling then this is is a float which is the weight
        given to each subtree

    s: int
        0 if sufficient leapfrog steps have been taken, 1 otherwise

    theta: ndarray
        the current accepted point along the path

    L: float
        the logpdf of the current accepted point

    grad_L: float
        the gradient of the logpdf at the current accepted point

    alpha: float
        the acceptance probability

    n_alpha: float
        a count of the points along this path

    divergent: boolean
        True if one of the points in the tree was divergent

    """

    def __init__(self, theta, r, L, grad_L, n, s, alpha, n_alpha, divergent,
                 use_multinomial_sampling):
        self.theta_minus = theta
        self.theta_plus = theta
        self.r_minus = r
        self.r_plus = r
        self.L_minus = L
        self.L_plus = L
        self.grad_L_minus = grad_L
        self.grad_L_plus = grad_L
        self.n = n
        self.s = s
        self.theta = theta
        self.L = L
        self.grad_L = grad_L
        self.alpha = alpha
        self.n_alpha = n_alpha
        self.divergent = divergent

        if use_multinomial_sampling:
            self.accumulate_weight = self.accumulate_multinomial_sampling
            self.probability_of_accept = \
                self.probability_of_accept_multinomial_sampling
        else:
            self.accumulate_weight = self.accumulate_slice_sampling
            self.probability_of_accept = \
                self.probability_of_accept_slice_sampling

    # for multinomial_sampling n will be a float, need to take care of
    # overflow when suming this log probability.
    def accumulate_multinomial_sampling(self, left_subtree, right_subtree):
        return np.logaddexp(left_subtree, right_subtree)

    def accumulate_slice_sampling(self, left_subtree, right_subtree):
        return left_subtree + right_subtree

    def probability_of_accept_multinomial_sampling(self, tree_n, subtree_n):
        return np.exp(subtree_n - tree_n)

    def probability_of_accept_slice_sampling(self, tree_n, subtree_n):
        return subtree_n / tree_n

    def update(self, other_state, direction, root):
        """
        if ``root == True``, this combines a depth j subtree (``self``) with a
        depth j+1 (``other_state``) subtree, which corresponds to the higher
        level loop in the nuts algorithm

        if ``root == False``, this combins two subtrees with depth j, which
        occurs when the nuts algorithm is implicitly building up the tree with
        the build_tree subroutine

        direction is the current direction of integration, either forwards
        (``direction == 1``), or backwards (``direction = -1``)
        """
        if direction == -1:
            self.theta_minus = other_state.theta_minus
            self.r_minus = other_state.r_minus
            self.L_minus = other_state.L_minus
            self.grad_L_minus = other_state.grad_L_minus
        else:
            self.theta_plus = other_state.theta_plus
            self.r_plus = other_state.r_plus
            self.L_plus = other_state.L_plus
            self.grad_L_plus = other_state.grad_L_plus

        theta_dash = other_state.theta
        L_dash = other_state.L
        grad_L_dash = other_state.grad_L
        n_dash = other_state.n
        s_dash = other_state.s
        alpha_dash = other_state.alpha
        n_alpha_dash = other_state.n_alpha

        # for non-root merges accumulate tree weightings before probability
        # calculation
        if not root:
            self.n = self.accumulate_weight(self.n, n_dash)

        # if there is any accepted points in the other subtree then test for
        # acceptance of that subtree's theta
        # probability of sample being in new tree only greater than 0 if
        # ``s_dash == 1``.  for non-root we don't need to check this as the new
        # tree is not built at all when ``s_dash != 1``
        if root:
            p = int(s_dash == 1) \
                * min(1, self.probability_of_accept(self.n, n_dash))
        else:
            p = self.probability_of_accept(self.n, n_dash)

        if p > 0.0 and np.random.uniform() < p:
            self.theta = theta_dash
            self.L = L_dash
            self.grad_L = grad_L_dash

        # for root merges accumulate tree weightings after probability
        # calculation
        if root:
            self.n = self.accumulate_weight(self.n, n_dash)

        # Notes: alpha and n_alpha are only accumulated within build_tree
        if root:
            self.alpha = alpha_dash
            self.n_alpha = n_alpha_dash
        else:
            self.alpha += alpha_dash
            self.n_alpha += n_alpha_dash

        # test if the path has done a U-Turn
        self.s *= s_dash
        self.s *= int((self.theta_plus -
                       self.theta_minus).dot(self.r_minus) >= 0)
        self.s *= int((self.theta_plus -
                       self.theta_minus).dot(self.r_plus) >= 0)

        # propogate divergence up the tree
        self.divergent |= other_state.divergent


# All the functions below are written as coroutines to enable the recursive
# nuts algorithm to be written using the ask-and-tell interface used by PINTS,
# see main coroutine function ``nuts_sampler`` for more details

@asyncio.coroutine
def leapfrog(theta, L, grad_L, r, epsilon, step_size):
    """ performs a leapfrog step with step size ``epsilon*step_size`` """
    r_new = r + 0.5 * epsilon * step_size * grad_L
    theta_new = theta + epsilon * step_size * r_new
    L_new, grad_L_new = (yield theta_new)
    r_new += 0.5 * epsilon * step_size * grad_L_new
    return L_new, grad_L_new, theta_new, r_new


@asyncio.coroutine
def build_tree(state, log_u, v, j, epsilon, hamiltonian0, step_size,
               hamiltonian_threshold, use_multinomial_sampling):
    """
    Implicitly build up a subtree of depth j for the NUTS sampler
    """
    if j == 0:
        # Base case - take one leapfrog in the direction v
        if v == -1:
            theta = state.theta_minus
            r = state.r_minus
            L = state.L_minus
            grad_L = state.grad_L_minus
        else:
            theta = state.theta_plus
            r = state.r_plus
            L = state.L_plus
            grad_L = state.grad_L_plus

        L_dash, grad_L_dash, theta_dash, r_dash = \
            yield from leapfrog(theta, L, grad_L, r, v * epsilon, step_size)
        hamiltonian_dash = L_dash - 0.5 * r_dash.dot(r_dash)
        slice_comparison = log_u - hamiltonian_dash
        if use_multinomial_sampling:
            n_dash = -slice_comparison
        else:
            n_dash = int(slice_comparison <= 0)
        comparison = hamiltonian_dash - hamiltonian0
        s_dash = int(slice_comparison < hamiltonian_threshold)
        divergent = not bool(s_dash)
        alpha_dash = min(1.0, np.exp(comparison))
        n_alpha_dash = 1
        return NutsState(
            theta_dash, r_dash, L_dash, grad_L_dash, n_dash, s_dash,
            alpha_dash, n_alpha_dash, divergent, use_multinomial_sampling
        )

    else:
        # Recursion - implicitly build the left and right subtrees
        state_dash = yield from  \
            build_tree(state, log_u, v, j - 1, epsilon, hamiltonian0,
                       step_size, hamiltonian_threshold,
                       use_multinomial_sampling)

        if state_dash.s == 1:
            state_double_dash = yield from \
                build_tree(state_dash, log_u, v, j - 1, epsilon, hamiltonian0,
                           step_size, hamiltonian_threshold,
                           use_multinomial_sampling)
            state_dash.update(state_double_dash, direction=v, root=False)

        return state_dash


@asyncio.coroutine
def find_reasonable_epsilon(theta, L, grad_L, step_size):
    """
    Pick a reasonable value of epsilon close to when the acceptance
    probability of the Langevin proposal crosses 0.5.
    """

    # intialise at epsilon = 1.0 (shouldn't matter where we start)
    epsilon = 1.0
    r = np.random.normal(size=len(theta))
    hamiltonian = L - 0.5 * r.dot(r)
    L_dash, grad_L_dash, theta_dash, r_dash = \
        yield from leapfrog(theta, L, grad_L, r, epsilon, step_size)
    hamiltonian_dash = L_dash - 0.5 * r_dash.dot(r_dash)
    comparison = hamiltonian_dash - hamiltonian

    # determine whether we are doubling or halving
    alpha = 2 * int(comparison > np.log(0.5)) - 1

    # double or half epsilon until acceptance probability crosses 0.5
    while comparison * alpha > np.log(2) * (-alpha):
        epsilon = 2**alpha * epsilon
        L_dash, grad_L_dash, theta_dash, r_dash = \
            yield from leapfrog(theta, L, grad_L, r, epsilon, step_size)
        hamiltonian_dash = L_dash - 0.5 * r_dash.dot(r_dash)
        comparison = hamiltonian_dash - hamiltonian
    return epsilon


@asyncio.coroutine
def nuts_sampler(x0, delta, M_adapt, step_size,
                 hamiltonian_threshold, max_tree_depth,
                 use_multinomial_sampling):
    """
    The dual averaging NUTS mcmc sampler given in Algorithm 6 of [1].

    Implemented as a coroutine that continually generates new theta values to
    evaluate (L, L') at. Users must send (L, L') back to the coroutine to
    continue execution. The end of an mcmc step is signalled by generating a
    tuple of values (theta, L, acceptance probability, number of leapfrog
    steps)

    Arguments
    ---------

    x0: ndarray
        starting point
    delta: float
        target acceptance probability (Dual Averaging scheme)
    M_adapt: int
        number of adaption steps (Dual Averaging scheme)
    step_size: ndarray
        step sizes in each parameter dimension
    hamiltonian_threshold: float
        threshold to test divergent iterations
    max_tree_depth: int
        maximum tree depth
    use_multinomial_sampling: bool
        use multinomial sampling as suggested in [2] instead of slice sampling
        method used in [1]

    References
    ----------
    .. [1] Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn sampler:
           adaptively setting path lengths in Hamiltonian Monte Carlo.
           Journal of Machine Learning Research, 15(1), 1593-1623.

    .. [2] `A Conceptual Introduction to Hamiltonian Monte Carlo`,
            Michael Betancourt

    """
    # Initialise sampler with x0 and calculate logpdf
    theta = x0
    L, grad_L = (yield theta)

    # Check first point is somewhere sensible
    if not np.isfinite(L):
        raise ValueError(
            'Initial point for MCMC must have finite logpdf.')

    # find a good value to start epsilon at
    # (this will later be refined so that the acceptance probability matches
    # delta)
    epsilon = yield from find_reasonable_epsilon(theta, L, grad_L, step_size)

    # default values taken from [1]
    mu = np.log(10 * epsilon)
    log_epsilon_bar = np.log(1)
    H_bar = 0
    gamma = 0.05
    t0 = 10
    kappa = 0.75

    # start at iteration 1
    m = 1

    # provide an infinite generator of mcmc steps....
    while True:
        # randomly sample momentum
        r0 = np.random.normal(size=len(theta))
        hamiltonian0 = L - 0.5 * r0.dot(r0)

        if use_multinomial_sampling:
            # use multinomial sampling
            log_u = hamiltonian0
            # n is a tree weight using a float
            n = 0.0
        else:
            # use slice sampling
            log_u = np.log(np.random.uniform(0, 1)) + hamiltonian0
            # n is the number of accepted points in the chain
            n = 1

        # create initial integration path state
        state = NutsState(theta, r0, L, grad_L, n, 1, None, None, False,
                use_multinomial_sampling)
        j = 0

        # build up an integration path with 2^j points, stopping when we either
        # encounter a U-Turn, or reach a max number of points 2^10
        while j < max_tree_depth and state.s == 1:

            # pick a random direction to integrate in
            # (to maintain detailed balance)
            if np.random.randint(0, 2):
                vj = 1
            else:
                vj = -1

            # recursivly build up tree in that direction
            state_dash = yield from \
                build_tree(state, log_u, vj, j, epsilon,
                           hamiltonian0, step_size, hamiltonian_threshold,
                           use_multinomial_sampling)
            state.update(state_dash, direction=vj, root=True)

            j += 1

        # adapt epsilon using dual averaging
        if m < M_adapt:
            H_bar = (1 - 1.0 / (m + t0)) * H_bar + 1.0 / (m + t0) * \
                (delta - state.alpha / state.n_alpha)
            log_epsilon = mu - (np.sqrt(m) / gamma) * H_bar
            log_epsilon_bar = m**(-kappa) * log_epsilon + \
                (1 - m**(-kappa)) * log_epsilon_bar
            epsilon = np.exp(log_epsilon)
        elif m == M_adapt:
            epsilon = np.exp(log_epsilon_bar)

        # update current position in chain
        theta = state.theta
        L = state.L
        grad_L = state.grad_L

        # signal calling process that mcmc step is complete by passing a tuple
        # (rather than an ndarray)
        yield (theta, L, state.alpha / state.n_alpha, 2**j, state.divergent)

        # next step
        m += 1


class NoUTurnMCMC(pints.SingleChainMCMC):
    r"""

    Implements No U-Turn Sampler (NUTS) with dual averaging, as described in
    Algorithm 6 in [1]_. Like Hamiltonian Monte Carlo, NUTS imagines a particle
    moving over negative log-posterior (NLP) space to generate proposals.
    Naturally, the particle tends to move to locations of low NLP -- meaning
    high posterior density. Unlike HMC, NUTS allows the number of steps taken
    through parameter space to depend on position, allowing local adaptation.

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
        self._M_adapt = 500
        self._delta = 0.6
        self._step_size = None
        self.set_leapfrog_step_size(np.diag(self._sigma0))
        self._max_tree_depth = 10
        self._use_multinomial_sampling = True

        # Default threshold for Hamiltonian divergences
        # (currently set to match Stan)
        self._hamiltonian_threshold = 10**3

        # coroutine nuts sampler
        self._nuts = None

        # number of mcmc iterations
        self._mcmc_iteration = 0

        # Logging
        self._last_log_write = 0
        self._mcmc_acceptance = 0
        self._n_leapfrog = 0

        # current point in chain
        self._current = self._x0

        # current logpdf (last logpdf returned by tell)
        self._current_logpdf = None

        # next point to ask user to evaluate
        self._next = self._current

        # Set initial state
        self._running = False
        self._ready_for_tell = False

        # Divergence checking
        # Create a vector of divergent iterations
        self._divergent = np.asarray([], dtype='int')

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """
        # Check ask/tell pattern
        if self._ready_for_tell:
            raise RuntimeError('Ask() called when expecting call to tell().')

        # Initialise on first call
        if not self._running:
            self._nuts = nuts_sampler(self._x0, self._delta, self._M_adapt,
                                      self._step_size,
                                      self._hamiltonian_threshold,
                                      self._max_tree_depth,
                                      self._use_multinomial_sampling)
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

        # send log likelihood and gradient to nuts coroutine,
        # return value is the next theta to evaluate at
        self._next = self._nuts.send(reply)

        # coroutine signals end of current step by sending a tuple of
        # information about the last mcmc step
        if isinstance(self._next, tuple):
            # extract next point in chain, its logpdf, the acceptance
            # probability and the number of leapfrog steps taken during
            # the last mcmc step
            self._current = self._next[0]
            self._current_logpdf = self._next[1]
            current_acceptance = self._next[2]
            current_n_leapfrog = self._next[3]
            divergent = self._next[4]

            # Increase iteration count
            self._mcmc_iteration += 1

            # average quantities for logging
            n_it_since_log = self._mcmc_iteration - self._last_log_write
            self._mcmc_acceptance = (
                (n_it_since_log * self._mcmc_acceptance + current_acceptance) /
                (n_it_since_log + 1)
            )
            self._n_leapfrog = (
                (n_it_since_log * self._n_leapfrog + current_n_leapfrog) /
                (n_it_since_log + 1)
            )

            # store divergent iterations
            if divergent:
                self._divergent = np.append(
                    self._divergent, self._mcmc_iteration)

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
        logger.add_counter('Steps.')

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        # print nothing if no mcmc iterations since last log
        if self._last_log_write == self._mcmc_iteration:
            logger.log(None)
            logger.log(None)
        else:
            logger.log(self._mcmc_acceptance)
            logger.log(self._n_leapfrog)
        self._mcmc_acceptance = 0
        self._n_leapfrog = 0
        self._last_log_write = self._mcmc_iteration

    def current_log_pdf(self):
        """ See :meth:`SingleChainMCMC.current_log_pdf()`. """
        return self._current_logpdf

    def hamiltonian_threshold(self):
        """
        Returns threshold difference in Hamiltonian value from one iteration to
        next which determines whether an iteration is divergent.
        """
        return self._hamiltonian_threshold

    def set_hamiltonian_threshold(self, hamiltonian_threshold):
        """
        Sets threshold difference in Hamiltonian value from one iteration to
        next which determines whether an iteration is divergent.
        """
        if hamiltonian_threshold < 0:
            raise ValueError('Threshold for divergent iterations must be ' +
                             'non-negative.')
        self._hamiltonian_threshold = hamiltonian_threshold

    def max_tree_depth(self):
        """
        Returns the maximum tree depth ``D`` for the algorithm. For each
        iteration, the number of leapfrog steps will not be greater than
        ``2^D``
        """
        return self._max_tree_depth

    def set_max_tree_depth(self, max_tree_depth):
        """
        Sets the maximum tree depth ``D`` for the algorithm. For each
        iteration, the number of leapfrog steps will not be greater than
        ``2^D``
        """
        if max_tree_depth < 0:
            raise ValueError('Maximum tree depth must be non-negative.')
        self._max_tree_depth = max_tree_depth

    def divergent_iterations(self):
        """
        Returns the iteration number of any divergent iterations
        """
        return self._divergent

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
        return 4

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'No-U-Turn MCMC'

    def needs_sensitivities(self):
        """ See :meth:`pints.MCMCSampler.needs_sensitivities()`. """
        return True

    def set_leapfrog_step_size(self, step_size):
        """
        Sets the step size for the leapfrog algorithm. Note that the absolute
        value of the step size is unimportant, as it will be scaled by a scalar
        step size that is generated by the dual averaging algorithm. It is
        important however to specify the correct ratio of step size between
        parameter dimensions
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

    def set_delta(self, delta):
        """
        Sets delta for the nuts algorithm. This is the goal acceptance
        probability for the algorithm. Used to set the scalar magnitude of the
        leapfrog step size
        """
        if self._running:
            raise RuntimeError('cannot set delta while sampler is running')
        if delta < 0 or delta > 1:
            raise ValueError('delta must be in [0, 1]')
        self._delta = delta

    def set_number_adaption_steps(self, n):
        """
        Sets number of adaptions steps in the nuts algorithm. This is the
        number of mcmc steps that are used to determin the best value for
        epsilon, the scalar magnitude of the leafrog step size
        """
        if self._running:
            raise RuntimeError(
                'cannot set number of adaption steps while sampler is running')
        if n < 0:
            raise ValueError('number of adaption steps must be non-negative')
        self._M_adapt = int(n)

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[delta, number_adaption_steps,
        leapfrog_step_size, max_tree_depth]``.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_delta(x[0])
        self.set_number_adaption_steps(x[1])
        self.set_leapfrog_step_size(x[2])
        self.set_max_tree_depth(x[3])
