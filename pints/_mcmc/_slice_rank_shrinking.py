#
# Slice Sampling - Covariance Adaptive: Rank Shrinking
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import numpy as np


class SliceRankShrinkingMCMC(pints.SingleChainMCMC):
    r"""
    Implements Covariance-Adaptive slice sampling by "rank shrinking",
    as introduced in [1]_ with pseudocode given in Fig. 5.

    This is an adaptive multivariate method which uses additional points,
    called "crumbs", and rejected proposals to guide the selection of samples.

    It generates samples by sampling uniformly from the volume underneath the
    posterior (:math:`f`). It does so by introducing an auxiliary variable
    (:math:`y`) that guide the path of a Markov chain.

    Sampling follows:

    1. Calculate the pdf (:math:`f(x_0)`) of the current sample
    :math:`(x_0)`.
    2. Draw a real value (:math:`y`) uniformly from :math:`(0, f(x0))`,
    defining a horizontal "slice": :math:`S = {x: y < f(x)}`. Note that
    :math:`x_0` is always within :math:`S`.
    3. Draw the first crumb (:math:`c_1`) from a Gaussian distribution with
    mean :math:`x_0` and precision matrix :math:`W_1`.
    4. Draw a new point (:math:`x_1`) from a Gaussian distribution with mean
    :math:`c_1` and precision matrix :math:`W_2`.

    New crumbs are drawn until a new proposal is accepted. In particular,
    after sampling :math:`k` crumbs from Gaussian distributions with mean
    :math:`x0` and precision matrices :math:`(W_1, ..., W_k)`, the distribution
    for the kth proposal sample is:

    .. math::
        x_k \sim Normal(\bar{c}_k, \Lambda^{-1}_k)

    where:

       :math:`\Lambda_k = W_1 + ... + W_k`
       :math:`\bar{c}_k = \Lambda^{-1}_k * (W_1 * c_1 + ... + W_k * c_k)`

    This method aims to conveniently modify the (k+1)th proposal distribution
    to increase the likelihood of sampling an acceptable point. It does so by
    calculating the gradient (:math:`g(f(x))`) of the unnormalised posterior
    (:math:`f(x)`) at the last rejected point (:math:`x_k`). It then sets the
    conditional variance of the (k + 1)th proposal distribution in the
    direction of the gradient :math:`g(f(x_k))` to 0. This is reasonable in
    that the gradient at a proposal probably points in a direction where the
    variance is small, so it is more efficient to move in a different
    direction.

    To avoid floating-point underflow, we implement the suggestion advanced
    in [2]_ pp.712. We use the log pdf of the un-normalised posterior
    (:math:`\text{log} f(x)`) instead of :math:`f(x)`. In doing so, we use an
    auxiliary variable :math:`z = log(y) - \epsilon`, where
    :math:`\epsilon \sim \text{exp}(1)` and define the slice as
    :math:`S = {x : z < log f(x)}`.

    Extends :class:`SingleChainMCMC`.

    References
    ----------
    .. [1] "Covariance-Adaptive Slice Sampling", 2010, M Thompson and RM Neal,
           Technical Report No. 1002, Department of Statistics, University of
           Toronto
    .. [2] "Slice sampling", 2003, Neal, R.M., The annals of statistics, 31(3),
           pp.705-767. https://doi.org/10.1214/aos/1056562461
    """

    def __init__(self, x0, sigma0=None):
        super(SliceRankShrinkingMCMC, self).__init__(x0, sigma0)

        # Set initial state
        self._x0 = np.asarray(x0, dtype=float)
        self._running = False
        self._ready_for_tell = False
        self._current = None
        self._current_log_y = None
        self._proposed = None

        # Standard deviation of initial crumb
        self._sigma_c = 1

        # Matrix of orthonormal columns of directions in which the conditional
        # variance is zero
        self._J = np.zeros((self._n_parameters, 0), float)

        # Number of crumbs
        self._k = 0

        # Mean of unprojected proposal distribution
        self._c_bar = 0

    # Function returning the component of vector v orthogonal to the
    # columns of J
    def _p(self, J, v):
        if not J.any():
            return np.array(v, copy=True)
        else:
            return np.array(v - np.dot(J, np.dot(J.transpose(), v)), copy=True)

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

            # Ask for the log pdf of x0
            self._ready_for_tell = True
            return np.array(self._x0, copy=True)

        # Increase crumbs count
        self._k += 1

        # Sample crumb
        mean = np.zeros(self._n_parameters)
        cov = np.identity(self._n_parameters)
        z = np.random.multivariate_normal(mean, cov)
        c = self._sigma_c * z

        # Mean of proposal distribution
        self._c_bar = ((self._k - 1) * self._c_bar + c) / self._k

        # Sample trial point
        z = np.random.multivariate_normal(mean, cov)
        self._proposed = self._current + self._p(
            self._J, self._c_bar + self._sigma_c / np.sqrt(self._k) * z)

        # Send trial point for checks
        self._ready_for_tell = True
        return np.copy(self._proposed)

    def current_slice_height(self):
        """
        Returns the height of the current slice.
        """
        return self._current_log_y

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Slice Sampling - Covariance-Adaptive: Rank Shrinking.'

    def needs_sensitivities(self):
        """ See :meth:`pints.MCMCSampler.needs_sensitivities()`. """
        return True

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 1

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[sigma_c]``.
        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_sigma_c(x[0])

    def set_sigma_c(self, sigma_c):
        """
        Sets standard deviation of initial crumb distribution.
        """
        sigma_c = float(sigma_c)
        if sigma_c < 0:
            raise ValueError(
                'Inital crumb standard deviation must be positive.')
        self._sigma_c = sigma_c

    def sigma_c(self):
        """
        Returns standard deviation of initial crumb distribution.
        """
        return self._sigma_c

    def tell(self, reply):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """

        # Check ask/tell pattern
        if not self._ready_for_tell:
            raise RuntimeError('Tell called before proposal was set.')
        self._ready_for_tell = False

        # Unpack reply
        fx, grad = reply

        # Check reply, copy gradient
        fx = float(fx)
        grad = pints.vector(grad)
        assert grad.shape == (self._n_parameters, )

        # Very first call
        if self._current is None:
            # Check first point is somewhere sensible
            if not np.isfinite(fx):
                raise ValueError(
                    'Initial point for MCMC must have finite logpdf.')

            # Set current sample, log pdf of current sample and initialise
            # proposed sample for next iteration
            self._current = np.array(self._x0, copy=True)
            self._proposed = np.array(self._current, copy=True)

            # Sample height of the slice log_y for next iteration
            e = np.random.exponential(1)
            self._current_log_y = fx - e

            # Return first point in chain, which is x0
            # Note: `grad` is not stored in this iteration, so can return
            return np.copy(self._current), (fx, grad), True

        # Acceptance check
        if fx >= self._current_log_y:
            # The accepted sample becomes the new current sample
            self._current = np.copy(self._proposed)

            # Sample new log_y used to define the next slice
            e = np.random.exponential(1)
            self._current_log_y = fx - e

            # Reset parameters
            self._J = np.zeros((self._n_parameters, 0), float)
            self._k = 0
            self._c_bar = 0

            # Return accepted sample
            # Note: `grad` is not stored in this iteration, so can return
            return np.copy(self._current), (fx, grad), True

        # If proposal is reject, shrink rank of the next proposal distribution
        # by adding new orthonormal column to ``J``. This will represent a new
        # direction in which the conditional covariance of the proposal
        # distribution will be 0.
        else:
            # If J has less non-zero columns than``number of
            # parameters - 1``, shrink rank by adding new orthonormal
            # column

            if self._J.shape[1] < self._n_parameters - 1:
                # Gradient projection
                g_star = self._p(self._J, grad)

                # To prevent meaningless adaptations, we only perform this
                # operation when the angle between the gradient and its
                # projection into the nullspace of J is less than 60 degrees.
                if np.dot(g_star.transpose(), grad) > (
                        .5 * np.linalg.norm(g_star) * np.linalg.norm(grad)):
                    new_column = np.array(g_star / np.linalg.norm(g_star))
                    self._J = np.column_stack([self._J, new_column])
            return None
