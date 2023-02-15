#
# No-U-Turn Sampler (NUTS) MCMC method
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import _pickle as cPickle

import pints
import numpy as np


class NutsState(object):
    """
    Class to hold information about the current state of the NUTS hamiltonian
    integration path.

    NUTS builds up the integration path implicitly via recursion up a binary
    tree, this class handles combining states from different subtrees (see
    `update`). The algorithm integrates both backwards ("minus") and forwards
    ("plus") in time, so this state must keep track of both end points of the
    integration path.

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
        the weight given to each subtree

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
                 inv_mass_matrix):
        self.theta_minus = np.copy(theta)
        self.theta_plus = np.copy(theta)
        self.r_minus = np.copy(r)
        self.r_plus = np.copy(r)
        self.r_sum = np.copy(r)
        self.L_minus = np.copy(L)
        self.L_plus = np.copy(L)
        self.grad_L_minus = np.copy(grad_L)
        self.grad_L_plus = np.copy(grad_L)
        self.n = n
        self.s = s
        self.theta = np.copy(theta)
        self.L = L
        self.grad_L = np.copy(grad_L)
        self.alpha = alpha
        self.n_alpha = n_alpha
        self.divergent = divergent
        self.inv_mass_matrix = inv_mass_matrix

    def update(self, other_state, direction, root):
        """
        if ``root == True``, this combines a depth j subtree (``self``) with a
        depth j+1 (``other_state``) subtree, which corresponds to the higher
        level loop in the nuts algorithm.

        if ``root == False``, this combines two subtrees with depth j, which
        occurs when the nuts algorithm is implicitly building up the tree with
        the build_tree subroutine.

        direction is the current direction of integration, either forwards
        (``direction == 1``), or backwards (``direction = -1``).
        """

        # update the appropriate end of the tree according to what direction we
        # are integrating
        if direction == -1:
            self.theta_minus = other_state.theta_minus
            self.r_minus = other_state.r_minus
            r_minus_plus = other_state.r_plus
            r_plus_minus = self.r_minus
            r_sum_minus = other_state.r_sum
            r_sum_plus = self.r_sum
            self.L_minus = other_state.L_minus
            self.grad_L_minus = other_state.grad_L_minus
        else:
            self.theta_plus = other_state.theta_plus
            self.r_plus = other_state.r_plus
            r_minus_plus = self.r_plus
            r_plus_minus = other_state.r_minus
            r_sum_minus = self.r_sum
            r_sum_plus = other_state.r_sum
            self.L_plus = other_state.L_plus
            self.grad_L_plus = other_state.grad_L_plus

        # Notes: alpha and n_alpha are only accumulated within build_tree
        # Update: perhaps not according to stan code...
        if root:
            self.alpha += other_state.alpha
            self.n_alpha += other_state.n_alpha
        else:
            self.alpha += other_state.alpha
            self.n_alpha += other_state.n_alpha

        # propogate divergence up the tree
        self.divergent |= other_state.divergent

        self.s *= other_state.s

        # check if chain is stopping
        if self.s == 0:
            return

        # for non-root merges accumulate tree weightings before probability
        # calculation
        if not root:
            self.n = np.logaddexp(self.n, other_state.n)

        # accept a new point based on the weighting of the two trees
        p = min(1, np.exp(other_state.n - self.n))
        if p > 0.0 and np.random.uniform() < p:
            self.theta = other_state.theta
            self.L = other_state.L
            self.grad_L = other_state.grad_L

        # for root merges accumulate tree weightings after probability
        # calculation
        if root:
            self.n = np.logaddexp(self.n, other_state.n)

        # integrate momentum over chain
        self.r_sum += other_state.r_sum

        # test if the path has done a U-Turn, if we are stopping due to a
        # U-turn or a divergent iteration propogate this up the tree with
        # self.s
        if self.inv_mass_matrix.ndim == 1:
            r_sharp_minus = self.inv_mass_matrix * self.r_minus
            r_sharp_plus = self.inv_mass_matrix * self.r_plus
            r_sharp_plus_minus = self.inv_mass_matrix * r_plus_minus
            r_sharp_minus_plus = self.inv_mass_matrix * r_minus_plus
        else:
            r_sharp_minus = self.inv_mass_matrix.dot(self.r_minus)
            r_sharp_plus = self.inv_mass_matrix.dot(self.r_plus)
            r_sharp_plus_minus = self.inv_mass_matrix.dot(r_plus_minus)
            r_sharp_minus_plus = self.inv_mass_matrix.dot(r_minus_plus)

        # test merged trees
        self.s *= int((self.r_sum).dot(r_sharp_minus) > 0)
        self.s *= int((self.r_sum).dot(r_sharp_plus) > 0)

        # test across subtrees
        self.s *= int((r_sum_minus + r_plus_minus).dot(r_sharp_minus) > 0)
        self.s *= int((r_sum_minus + r_plus_minus).dot(r_sharp_plus_minus) > 0)

        self.s *= int((r_sum_plus + r_minus_plus).dot(r_sharp_minus_plus) > 0)
        self.s *= int((r_sum_plus + r_minus_plus).dot(r_sharp_plus) > 0)


def kinetic_energy(r, inv_mass_matrix):
    if inv_mass_matrix.ndim == 1:
        return 0.5 * np.inner(r, inv_mass_matrix * r)
    else:
        return 0.5 * np.inner(r, inv_mass_matrix.dot(r))

# All the functions below are written as coroutines to enable the recursive
# nuts algorithm to be written using the ask-and-tell interface used by PINTS,
# see main coroutine function ``nuts_sampler`` for more details


def leapfrog(theta, L, grad_L, r, epsilon, inv_mass_matrix):
    """
    performs a leapfrog step using a step_size ``epsilon`` and an inverse mass
    matrix ``inv_mass_matrix``.

    The inverse mass matrix can be a 2 dimensional ndarray, in which case it is
    interpreted as a dense matrix, or a 1 dimensional ndarray, in which case it
    is interpreted as a diagonal matrix.
    """
    r_new = r + 0.5 * epsilon * grad_L
    if inv_mass_matrix.ndim == 1:
        theta_new = theta + epsilon * inv_mass_matrix * r_new
    else:
        theta_new = theta + epsilon * inv_mass_matrix.dot(r_new)
    L_new, grad_L_new = (yield theta_new)
    r_new += 0.5 * epsilon * grad_L_new
    return L_new, grad_L_new, theta_new, r_new


def build_tree(state, v, j, adaptor, hamiltonian0, hamiltonian_threshold):
    """
    Implicitly build up a subtree of depth ``j`` for the NUTS sampler.
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
            yield from leapfrog(theta, L, grad_L, r, v * adaptor.get_epsilon(),
                                adaptor.get_inv_mass_matrix())

        hamiltonian_dash = L_dash \
            - kinetic_energy(r_dash, adaptor.get_inv_mass_matrix())

        if np.isnan(hamiltonian_dash):
            comparison = -np.inf
        else:
            comparison = hamiltonian_dash - hamiltonian0
        n_dash = comparison
        alpha_dash = min(1.0, np.exp(comparison))
        divergent = -comparison > hamiltonian_threshold
        s_dash = int(not divergent)
        n_alpha_dash = 1

        return NutsState(
            theta_dash, r_dash, L_dash, grad_L_dash, n_dash, s_dash,
            alpha_dash, n_alpha_dash, divergent,
            adaptor.get_inv_mass_matrix()
        )

    else:
        # Recursion - implicitly build the left and right subtrees
        state_dash = yield from  \
            build_tree(state, v, j - 1, adaptor, hamiltonian0,
                       hamiltonian_threshold)

        if state_dash.s == 1:
            state_double_dash = yield from \
                build_tree(state_dash, v, j - 1, adaptor, hamiltonian0,
                           hamiltonian_threshold)
            state_dash.update(state_double_dash, direction=v, root=False)

        return state_dash


def initialise_adaptor(
        theta, L, grad_L, num_adaption_steps, delta, sigma0,
        use_dense_mass_matrix):
    """
    Creates a generator that terminates by returning an instance of the
    pints.DualAveragingAdaption.

    Initialisation of the adaptor requires a 'reasonable' epsilon which
    is in turn also a generator. The find_reasonable_epsilon generator
    terminates with return of a 'reasonable' epsilon. Intermediate returns
    are the current position of the leapfrog integrator.
    """

    # pick the initial inverse mass matrix as the provided sigma0.
    # reduce to a diagonal matrix if not using a dense mass matrix
    if use_dense_mass_matrix:
        init_inv_mass_matrix = sigma0
        init_inv_mass_matrix = 1e-3 * np.eye(len(theta))
    else:
        init_inv_mass_matrix = np.diag(sigma0)
        init_inv_mass_matrix = 1e-3 * np.ones(len(theta))

    # find a good value to start epsilon at (this will later be refined so that
    # the acceptance probability matches delta)
    epsilon = yield from find_reasonable_epsilon(
        theta, L, grad_L, init_inv_mass_matrix)

    # create adaption for epsilon and mass matrix
    return pints.DualAveragingAdaption(
        num_adaption_steps, delta, epsilon, init_inv_mass_matrix)


def find_reasonable_epsilon(theta, L, grad_L, inv_mass_matrix):
    """
    Pick a reasonable value of epsilon close to when the acceptance
    probability of the Langevin proposal crosses 0.5. This is based on
    Algorithm 4 in [1]_ (with scaled mass matrix as per section 4.2).

    Note: inv_mass_matrix can be a 1-d ndarray and in this case is interpreted
    as a diagonal matrix, or can be given as a fully dense 2-d ndarray.
    """

    # intialise at epsilon = 1.0 (shouldn't matter where we start)
    epsilon = 1.0

    # randomly sample momentum
    if inv_mass_matrix.ndim == 1:
        r = np.random.normal(
            np.zeros(len(theta)),
            np.sqrt(1.0 / inv_mass_matrix)
        )
    else:
        r = np.random.multivariate_normal(
            np.zeros(len(theta)),
            np.linalg.inv(inv_mass_matrix)
        )
    hamiltonian = L - kinetic_energy(r, inv_mass_matrix)
    L_dash, grad_L_dash, theta_dash, r_dash = \
        yield from leapfrog(theta, L, grad_L, r, epsilon, inv_mass_matrix)
    hamiltonian_dash = L_dash - kinetic_energy(r_dash, inv_mass_matrix)
    if np.isnan(hamiltonian_dash):
        comparison = -np.inf
    else:
        comparison = hamiltonian_dash - hamiltonian

    # determine whether we are doubling or halving
    alpha = 2 * int(comparison > np.log(0.5)) - 1

    # double or half epsilon until acceptance probability crosses 0.5
    while comparison * alpha > np.log(2) * (-alpha):
        epsilon = 2**alpha * epsilon
        L_dash, grad_L_dash, theta_dash, r_dash = \
            yield from leapfrog(theta, L, grad_L, r, epsilon, inv_mass_matrix)
        hamiltonian_dash = L_dash - kinetic_energy(r_dash, inv_mass_matrix)
        if np.isnan(hamiltonian_dash):  # pragma: no cover
            comparison = -np.inf
        else:
            comparison = hamiltonian_dash - hamiltonian
    return epsilon


def nuts_sampler(
        x0, adaptor, sigma0, hamiltonian_threshold, max_tree_depth):
    """
    The dual averaging NUTS mcmc sampler given in Algorithm 6 of [1]_.
    Implements the multinomial sampling suggested in [2]_. Implements a mass
    matrix for the dynamics, which is detailed in [2]_. Both the step size and
    the mass matrix is adapted using a combination of the dual averaging
    detailed in [1]_ and the windowed adaption for the mass matrix and step
    size implemented in the Stan library (https://github.com/stan-dev/stan)

    Implemented as a coroutine that continually generates new theta values to
    evaluate (L, L') at. Users must send (L, L') back to the coroutine to
    continue execution. The end of an mcmc step is signalled by generating a
    tuple of values (theta, L, acceptance probability, number of leapfrog
    steps)

    Arguments
    ---------
    x0: ndarray
        starting point
    adaptor: list or pints.DualAveragingAdaption
        list with properties of the averaging algorithm [num_adaption_steps,
        delta, use_dense_mass_matrix] or instance of averaging algorithm.
    hamiltonian_threshold: float
        threshold to test divergent iterations
    max_tree_depth: int
        maximum tree depth

    References
    ----------
    .. [1] Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn sampler:
           adaptively setting path lengths in Hamiltonian Monte Carlo.
           Journal of Machine Learning Research, 15(1), 1593-1623.

    .. [2] Betancourt, M. (2018). `A Conceptual Introduction to Hamiltonian
           Monte Carlo`, https://arxiv.org/abs/1701.02434.

    """
    # Initialise sampler with x0 and calculate logpdf
    theta = x0
    L, grad_L = (yield theta)

    # Check first point is somewhere sensible
    if not np.isfinite(L):
        raise ValueError(
            'Initial point for MCMC must have finite logpdf.')

    # If adaptor is not yet created, initialise it with the provided
    # properties.
    if isinstance(adaptor, list):
        # Adaptor does currently not exist and is only specified by a list
        # of properties.
        num_adaption_steps, delta, use_dense_mass_matrix = adaptor
        adaptor = yield from initialise_adaptor(
            theta, L, grad_L, num_adaption_steps, delta, sigma0,
            use_dense_mass_matrix)

    # provide an infinite generator of mcmc steps....
    while True:
        # randomly sample momentum
        if adaptor.use_dense_mass_matrix():
            r0 = np.random.multivariate_normal(
                np.zeros(len(theta)), adaptor.get_mass_matrix())
        else:
            r0 = np.random.normal(np.zeros(len(theta)),
                                  np.sqrt(adaptor.get_mass_matrix()))

        hamiltonian0 = L - kinetic_energy(r0, adaptor.get_inv_mass_matrix())

        # create initial integration path state
        state = NutsState(theta=theta, r=r0, L=L, grad_L=grad_L,
                          n=0.0, s=1, alpha=1, n_alpha=1, divergent=False,
                          inv_mass_matrix=adaptor.get_inv_mass_matrix())
        j = 0

        # build up an integration path with 2^j points, stopping when we either
        # encounter a U-Turn, or reach a max number of points 2^max_tree_depth
        while j < max_tree_depth and state.s == 1:
            # pick a random direction to integrate in
            # (to maintain detailed balance)
            if np.random.randint(0, 2):
                vj = 1
            else:
                vj = -1

            # recursivly build up tree in that direction
            state_dash = yield from \
                build_tree(state, vj, j, adaptor,
                           hamiltonian0, hamiltonian_threshold)
            state.update(state_dash, direction=vj, root=True)

            j += 1

        # update current position in chain
        theta = state.theta
        L = state.L
        grad_L = state.grad_L

        # adapt epsilon and mass matrix using dual averaging
        restart_stepsize_adapt = \
            adaptor.step(state.theta, state.alpha / state.n_alpha)
        if restart_stepsize_adapt:
            epsilon = yield from \
                find_reasonable_epsilon(theta, L, grad_L,
                                        adaptor.get_inv_mass_matrix())
            adaptor.init_adapt_epsilon(epsilon)

        # signal calling process that mcmc step is complete by passing a tuple
        # (rather than an ndarray)
        yield (theta,
               L,
               grad_L,
               state.alpha / state.n_alpha,
               state.n_alpha,
               state.divergent,
               adaptor)


class NoUTurnMCMC(pints.SingleChainMCMC):
    r"""

    Implements the No U-Turn Sampler (NUTS) with dual averaging, as described
    in Algorithm 6 in [1]_.

    Implements the multinomial sampling suggested in [2]_. Implements a mass
    matrix for the dynamics, which is detailed in [2]_. Both the step size and
    the mass matrix is adapted using a combination of the dual averaging
    detailed in [1]_, and the windowed adaption for the mass matrix and step
    size implemented in the Stan library (https://github.com/stan-dev/stan).

    Like Hamiltonian Monte Carlo, NUTS imagines a particle moving over negative
    log-posterior (NLP) space to generate proposals. Naturally, the particle
    tends to move to locations of low NLP -- meaning high posterior density.
    Unlike HMC, NUTS allows the number of steps taken through parameter space
    to depend on position, allowing local adaptation.

    Note: This sampler is only supported on Python versions 3.3 and newer.

    Extends :class:`SingleChainMCMC`.

    References
    ----------
    .. [1] Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn sampler:
           adaptively setting path lengths in Hamiltonian Monte Carlo.
           Journal of Machine Learning Research, 15(1), 1593-1623.

    .. [2] Betancourt, M. (2018). `A Conceptual Introduction to Hamiltonian
           Monte Carlo`, https://arxiv.org/abs/1701.02434.

    """

    def __init__(self, x0, sigma0=None):
        super(NoUTurnMCMC, self).__init__(x0, sigma0)

        # hyperparameters
        self._adaptor = [
            500,           # Number of adaption steps
            0.8,           # Target acceptance ratio (delta)
            False,         # Uses dense mass matrix
        ]
        self._max_tree_depth = 10

        # Default threshold for Hamiltonian divergences
        # (currently set to match Stan)
        self._hamiltonian_threshold = 10**3

        # coroutine nuts sampler
        self._nuts = None
        self._nuts_state = None

        # number of mcmc iterations
        self._mcmc_iteration = 0

        # Logging
        self._last_log_write = 0
        self._mcmc_acceptance = 0
        self._n_leapfrog = 0

        # current point in chain
        self._current = self._x0

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
            self._nuts = nuts_sampler(
                self._x0, self._adaptor, self._sigma0,
                self._hamiltonian_threshold, self._max_tree_depth)

            # coroutine will ask for self._x0
            self._next = next(self._nuts)
            self._running = True

        self._ready_for_tell = True
        return np.array(self._next, copy=True)

    def delta(self):
        """
        Returns delta used in leapfrog algorithm.
        """
        try:
            # When adaptor has not been initialised
            return self._adaptor[1]
        except TypeError:
            # Adaptor has been initialised
            return self._adaptor.target_accept_prob()

    def divergent_iterations(self):
        """
        Returns the iteration number of any divergent iterations.
        """
        return self._divergent

    def hamiltonian_threshold(self):
        """
        Returns threshold difference in Hamiltonian value from one iteration to
        next which determines whether an iteration is divergent.
        """
        return self._hamiltonian_threshold

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

    def load_state(self, file):
        """
        Loads sampler state from a .pickle file and returns sampler.

        Arguments
        ---------
        file: str
            Path to file.
        """
        with open(file, "rb") as input_file:
            method = cPickle.load(input_file)

        # Recreate NUTS sampler state
        method._nuts = nuts_sampler(
            method._current, method._adaptor, method._sigma0,
            method._hamiltonian_threshold, method._max_tree_depth)

        # NOTE: This nuts_sampler still differs from before pickling, because
        # before returning the mcmc proposal the nuts sampler starts the next
        # leapfrog trajectory by sampling a new momentum and using the
        # current gradient information to perform the first leapfrog step. We
        # can't do this here, because we don't have the gradient information
        # anymore.
        # BUT we can prepare the ask method such that it returns the current
        # position, such that after one more ask-tell cycle we have caught up
        # with the sampler before pickling. Since this ask-tell cycle does not
        # return anything to the user, this effectively reconstructs the
        # sampler state. The momentum will however be resampled, which is ok
        # since it's the start of a new trajectory.
        method._next = next(method._nuts)

        return method

    def max_tree_depth(self):
        """
        Returns the maximum tree depth ``D`` for the algorithm. For each
        iteration, the number of leapfrog steps will not be greater than
        ``2^D``.
        """
        return self._max_tree_depth

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 1

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'No-U-Turn MCMC'

    def needs_sensitivities(self):
        """ See :meth:`pints.MCMCSampler.needs_sensitivities()`. """
        return True

    def number_adaption_steps(self):
        """
        Returns number of adaption steps used in the NUTS algorithm.
        """
        try:
            # When adaptor has not been initialised
            return self._adaptor[0]
        except TypeError:
            # Adaptor has been initialised
            return self._adaptor.warmup_steps()

    def save_state(self, file):
        """
        Saves sampler state to pickle file.

        Arguments
        ---------
        file: str
            Path to file.
        """
        # Remove nuts_sampler generator
        nuts = self._nuts
        self._nuts = None
        with open(file, "wb") as output_file:
            cPickle.dump(self, output_file)

        # Put generator back, in case the sampler is used further after
        # pickling
        self._nuts = nuts

    def set_delta(self, delta):
        """
        Sets delta for the nuts algorithm. This is the goal acceptance
        probability for the algorithm. Used to set the scalar magnitude of the
        leapfrog step size.
        """
        if self._running:
            raise RuntimeError('cannot set delta while sampler is running')
        if delta < 0 or delta > 1:
            raise ValueError('delta must be in [0, 1]')
        self._adaptor[1] = delta

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
        The hyper-parameter vector is ``[number_adaption_steps]``.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_number_adaption_steps(x[0])

    def set_max_tree_depth(self, max_tree_depth):
        """
        Sets the maximum tree depth ``D`` for the algorithm. For each
        iteration, the number of leapfrog steps will not be greater than
        ``2^D``
        """
        if max_tree_depth < 0:
            raise ValueError('Maximum tree depth must be non-negative.')
        self._max_tree_depth = max_tree_depth

    def set_number_adaption_steps(self, n):
        """
        Sets number of adaptions steps in the nuts algorithm. This is the
        number of mcmc steps that are used to determin the best value for
        epsilon, the scalar magnitude of the leafrog step size.
        """
        if self._running:
            raise RuntimeError(
                'cannot set number of adaption steps while sampler is running')
        if n < 0:
            raise ValueError('number of adaption steps must be non-negative')
        self._adaptor[0] = int(n)

    def set_use_dense_mass_matrix(self, use_dense_mass_matrix):
        """
        If ``use_dense_mass_matrix`` is False then algorithm uses a diagonal
        matrix for the mass matrix. If True then a fully dense mass matrix is
        used.
        """
        if self._running:
            raise RuntimeError(
                'cannot set number of adaption steps while sampler is running')
        self._adaptor[2] = bool(use_dense_mass_matrix)

    def tell(self, reply):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """
        if not self._ready_for_tell:
            raise RuntimeError('Tell called before proposal was set.')
        self._ready_for_tell = False

        # send log likelihood and gradient to nuts coroutine,
        # return value is the next theta to evaluate at but not necessarily the
        # proposed mcmc step. Final mcmc proposal is distinguished from
        # intermediate steps by tuple type.
        self._next = self._nuts.send(reply)

        # coroutine signals end of current step by sending a tuple of
        # information about the last mcmc step
        if isinstance(self._next, tuple):
            # extract next point in chain, its logpdf, the acceptance
            # probability and the number of leapfrog steps taken during
            # the last mcmc step
            self._current = self._next[0]
            current_logpdf = self._next[1]
            current_gradient = self._next[2]
            current_acceptance = self._next[3]
            current_n_leapfrog = self._next[4]
            divergent = self._next[5]
            self._adaptor = self._next[6]

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
            return (
                np.copy(self._current),
                (current_logpdf, np.copy(current_gradient)),
                True
            )
        else:
            # Return None to indicate there is no new sample for the chain
            return None

    def use_dense_mass_matrix(self):
        """
        Returns if the algorithm uses a dense (True) or diagonal (False) mass
        matrix.
        """
        try:
            # When adaptor has not been initialised
            return self._adaptor[2]
        except TypeError:
            # Adaptor has been initialised
            return self._adaptor.use_dense_mass_matrix()
