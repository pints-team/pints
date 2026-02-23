#
# Shared problems used in change point testing.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np

import pints
import pints.toy


class RunMcmcMethodOnProblem(object):
    """
    Base class for tests that run an MCMC method on a log-PDF.

    Parameters
    ----------
    log_pdf : pints.LogPDF
        The PDF to sample. Will be passed to a :class:`pints.MCMCController`.
    x0
        One or more starting points to be passed to the
        :class:`pints.MCMCController`.
    sigma0
        One or more ``sigma0`` parameters to be passed to the
        :class:`pints.MCMCController`.
    method : pints.MCMCSampler
        The method to test. Will be passed to the
        :class:`pints.MCMCController`.
    n_chains : int
        The number of chains to run. Will be passed to the
        :class:`pints.MCMCController`.
    n_iterations : int
        The number of iterations to run
    n_warmup : int
        The number of iterations to discard
    method_hyper_parameters : list
        A list of hyperparameter values.

    """

    def __init__(self, log_pdf, x0, sigma0, method, n_chains, n_iterations,
                 n_warmup, method_hyper_parameters):
        self.log_pdf = log_pdf

        controller = pints.MCMCController(
            log_pdf, n_chains, x0, sigma0=sigma0, method=method)
        controller.set_max_iterations(n_iterations)
        controller.set_log_to_screen(False)
        set_hyperparameters_for_any_mcmc_class(controller, method,
                                               method_hyper_parameters)
        self.chains = run_and_throw_away_warmup(controller, n_warmup)

    def estimate_kld(self):
        """
        Estimates the Kullback-Leibler divergence.

        Raises an ``AttributeError`` if the underlying LogPDF does not have a
        method ``kl_divergence()``.
        """
        chains = np.vstack(self.chains)
        return np.mean(self.log_pdf.kl_divergence(chains))

    def estimate_mean_ess(self):
        """
        Estimates the effective sample size (ESS) for each chain and each
        parameter and returns the mean ESS for across all chains and
        parameters.
        """
        # Estimate mean ESS for each chain
        n_chains, _, n_parameters = self.chains.shape
        ess = np.empty(shape=(n_chains, n_parameters))
        for chain_id, chain in enumerate(self.chains):
            ess[chain_id] = pints.effective_sample_size(chain)

        return np.mean(ess)

    def estimate_distance(self):
        """
        Estimates a measure of distance between the sampled chains and the true
        distribution.

        Raises an ``AttributeError`` if the underlying LogPDF does not have a
        method ``distance()``.
        """
        return self.log_pdf.distance(np.vstack(self.chains))


class RunMcmcMethodOnTwoDimGaussian(RunMcmcMethodOnProblem):
    """
    Tests a given MCMC method on a two-dimensional Gaussian distribution with
    means ``[0, 0]`` and sigma ``[1, 1]``.

    For constructor arguments, see :class:`RunMcmcMethodOnProblem`.
    """

    def __init__(self, method, n_chains, n_iterations, n_warmup,
                 method_hyper_parameters=None):
        log_pdf = pints.toy.GaussianLogPDF(mean=[0, 0], sigma=[1, 1])

        # Get initial parameters
        log_prior = pints.ComposedLogPrior(
            pints.GaussianLogPrior(mean=0, sd=10),
            pints.GaussianLogPrior(mean=0, sd=10))
        x0 = log_prior.sample(n=n_chains)
        sigma0 = None

        super().__init__(log_pdf, x0, sigma0, method, n_chains, n_iterations,
                         n_warmup, method_hyper_parameters)


class RunMcmcMethodOnBanana(RunMcmcMethodOnProblem):
    """
    Tests a given MCMC method on a two-dimensional
    :class:`pints.toy.TwistedGaussianLogPDF` distribution with means
    ``[0, 0]``.

    For constructor arguments, see :class:`RunMcmcMethodOnProblem`.
    """
    def __init__(self, method, n_chains, n_iterations, n_warmup,
                 method_hyper_parameters=None):
        log_pdf = pints.toy.TwistedGaussianLogPDF(dimension=2, b=0.1)

        # Get initial parameters
        log_prior = pints.MultivariateGaussianLogPrior([0, 0],
                                                       [[10, 0], [0, 10]])
        x0 = log_prior.sample(n_chains)
        sigma0 = np.diag(np.array([1, 3]))

        super().__init__(log_pdf, x0, sigma0, method, n_chains, n_iterations,
                         n_warmup, method_hyper_parameters)


'''
class RunMcmcMethodOnSimpleEggBox(RunMcmcMethodOnProblem):
    """
    Tests a given MCMC method on `pints.toy.SimpleEggBoxLogPDF`.
    """
    def __init__(self, method, n_chains, n_iterations, n_warmup,
                 method_hyper_parameters=None):
        sigma = 2
        r = 4
        log_pdf = pints.toy.SimpleEggBoxLogPDF(sigma, r)
        x0 = np.random.uniform(-15, 15, size=(n_chains, 2))
        sigma0 = None

        super().__init__(log_pdf, x0, sigma0, method, n_chains, n_iterations,
                         n_warmup, method_hyper_parameters)
'''


class RunMcmcMethodOnHighDimensionalGaussian(RunMcmcMethodOnProblem):
    """
    Tests a given MCMC method on a 20-dimensional
    :class:`pints.toy.HighDimensionalGaussianLogPDF` centered at the origin.

    For constructor arguments, see :class:`RunMcmcMethodOnProblem`.
    """
    def __init__(self, method, n_chains, n_iterations, n_warmup,
                 method_hyper_parameters=None):
        log_pdf = pints.toy.HighDimensionalGaussianLogPDF()
        x0 = np.random.uniform(-4, 4, size=(n_chains, 20))
        sigma0 = None

        super().__init__(log_pdf, x0, sigma0, method, n_chains, n_iterations,
                         n_warmup, method_hyper_parameters)


class RunMcmcMethodOnCorrelatedGaussian(RunMcmcMethodOnProblem):
    """
    Tests a given MCMC method on a 6-dimensional, highly correlated
    :class:`pints.toy.HighDimensionalGaussianLogPDF` centered at the origin.

    For constructor arguments, see :class:`RunMcmcMethodOnProblem`.
    """
    def __init__(self, method, n_chains, n_iterations, n_warmup,
                 method_hyper_parameters=None):
        log_pdf = pints.toy.HighDimensionalGaussianLogPDF(dimension=6, rho=0.8)
        x0 = np.random.uniform(-4, 4, size=(n_chains, 6))
        sigma0 = None

        super().__init__(log_pdf, x0, sigma0, method, n_chains, n_iterations,
                         n_warmup, method_hyper_parameters)


class RunMcmcMethodOnAnnulus(RunMcmcMethodOnProblem):
    """
    Tests a given MCMC method on a two-dimensional
    :class:`pints.toy.AnnulusLogPDF` distribution, with its highest values at
    any point ``x`` with ``np.linalg.norm(x) == 10``.

    For constructor arguments, see :class:`RunMcmcMethodOnProblem`.
    """
    def __init__(self, method, n_chains, n_iterations, n_warmup,
                 method_hyper_parameters=None):
        log_pdf = pints.toy.AnnulusLogPDF()
        x0 = log_pdf.sample(n_chains)
        sigma0 = None

        super().__init__(log_pdf, x0, sigma0, method, n_chains, n_iterations,
                         n_warmup, method_hyper_parameters)


class RunMcmcMethodOnMultimodalGaussian(RunMcmcMethodOnProblem):
    """
    Tests a given MCMC method on a two-dimensional
    :class:`pints.toy.MultimodalGaussianLogPDF` with modes at ``[0, 0]``,
    ``[5, 10]``, and ``[10, 0]``.

    For constructor arguments, see :class:`RunMcmcMethodOnProblem`.
    """
    def __init__(self, method, n_chains, n_iterations, n_warmup,
                 method_hyper_parameters=None):
        modes = [[0, 0],
                 [5, 10],
                 [10, 0]]
        covariances = [[[1, 0], [0, 1]],
                       [[2, 0.8], [0.8, 3]],
                       [[1, -0.5], [-0.5, 1]]]
        log_pdf = pints.toy.MultimodalGaussianLogPDF(modes, covariances)
        x0 = log_pdf.sample(n_chains)
        sigma0 = None

        super().__init__(log_pdf, x0, sigma0, method, n_chains, n_iterations,
                         n_warmup, method_hyper_parameters)


class RunMcmcMethodOnCone(RunMcmcMethodOnProblem):
    """
    Tests a given MCMC method on a two-dimensional
    :class:`pints.toy,ConeLogPDF` centered at ``[0, 0]``.

    For constructor arguments, see :class:`RunMcmcMethodOnProblem`.
    """
    def __init__(self, method, n_chains, n_iterations, n_warmup,
                 method_hyper_parameters=None):
        log_pdf = pints.toy.ConeLogPDF(dimensions=2, beta=0.6)
        x0 = log_pdf.sample(n_chains)
        sigma0 = None

        super().__init__(log_pdf, x0, sigma0, method, n_chains, n_iterations,
                         n_warmup, method_hyper_parameters)


class RunOptimiserOnProblem(object):
    """
    Base class for tests that run an optimiser on an error function or log-PDF.

    Parameters
    ----------
    error : pints.Error or pints.LogPDF
        The function to opimise. Will be passed to a
        :class:`pints.OptimisationController`.
    x0
        A starting point to be passed to the controller.
    sigma0
        An optional ``sigma0`` argument to pass to the controller.
    method : pints.Optimiser
        The method to test.
    n_iterations : int
        The number of iterations to run.
    use_guessed : bool
        Set to true to use ``f_guessed_tracking`` (see
        :meth:`Optimiser.set_f_guessed_tracking`).
    method_hyper_parameters : list
        A list of hyperparameter values.

    """

    def __init__(self, error, x0, sigma0, boundaries, transformation, method,
                 xtrue, n_iterations, use_guessed=False,
                 method_hyper_parameters=None):
        self._error = error
        self._xtrue = pints.vector(xtrue)

        controller = pints.OptimisationController(
            error, x0, sigma0, boundaries, transformation, method)
        controller.set_max_iterations(n_iterations)
        controller.set_max_unchanged_iterations(None)
        controller.set_log_to_screen(False)
        if use_guessed:
            controller.set_f_guessed_tracking(True)
        if method_hyper_parameters is not None:
            controller.optimiser().set_hyperparameters(method_hyper_parameters)
        self._x, self._f = controller.run()

    def distance(self):
        """
        Calculates the distance between the obtained solution and the true
        solution.
        """
        return np.sqrt(np.sum((self._x - self._xtrue)**2))

    def error(self):
        """
        Returns the final error.
        """
        return self._f


class RunOptimiserOnBoundedFitzhughNagumo(RunOptimiserOnProblem):
    """
    Tests a given Optimiser on a fully observable (multi-output)
    Fitzhugh-Nagumo model, using boundaries but no transformations (the scales
    of the parameters are relatively similar).
    """
    def __init__(self, method, n_iterations, use_guessed=False,
                 method_hyper_parameters=None):

        # Choose starting point. The loss surface does not suggest any sensible
        # way to do this, so just sampling in a small sphere around a chosen x.
        x0 = [0.75, 1.5, 3]                 # Center
        r = np.random.uniform(0, 0.2)       # Sphere radius
        t = np.random.uniform(0, 2 * np.pi)
        p = np.random.uniform(0, 2 * np.pi)
        x0[0] += r * np.sin(t) * np.cos(p)
        x0[1] += r * np.sin(t) * np.sin(p)
        x0[2] += r * np.cos(t)
        # Note that this is not a uniform sampling from the sphere!
        sigma0 = 0.05

        # Create a seeded generator to get consistent noise
        r = np.random.default_rng(1)

        # Create problem
        model = pints.toy.FitzhughNagumoModel()
        xtrue = model.suggested_parameters()
        times = model.suggested_times()
        values = model.simulate(xtrue, times)
        values += r.normal(0, 0.25, values.shape)
        problem = pints.MultiOutputProblem(model, times, values)
        error = pints.SumOfSquaresError(problem)

        # Add boundaries
        boundaries = pints.RectangularBoundaries(
            [1e-3, 1e-3, 1e-3], [2, 2, 10])

        super().__init__(error, x0, sigma0, boundaries, None, method, xtrue,
                         n_iterations, use_guessed, method_hyper_parameters)


class RunOptimiserOnBoundedUntransformedLogistic(RunOptimiserOnProblem):
    """
    Tests a given Optimiser on a logistic model inference problem with
    boundaries and very different scalings for the parameters (no sigma0
    information is given).
    """
    def __init__(self, method, n_iterations, use_guessed=False,
                 method_hyper_parameters=None):
        # Choose starting point
        # For the default parameters, the contours of the score function with
        # x[1] = 15 are almost horizontal after x[0] = 0.1, so we can fix Y and
        # vary X to get starting points with similar errors.
        x0 = np.array([np.random.uniform(0.15, 9), 15])

        # Create random generator to add consistent noise
        r = np.random.default_rng(1)

        # Create problem
        model = pints.toy.LogisticModel()
        xtrue = model.suggested_parameters()
        times = model.suggested_times()
        values = model.simulate(xtrue, times)
        values += r.normal(0, 5, values.shape)
        problem = pints.SingleOutputProblem(model, times, values)
        error = pints.SumOfSquaresError(problem)

        # Add boundaries
        boundaries = pints.RectangularBoundaries([0, 0.5], [10, 100])

        super().__init__(error, x0, None, boundaries, None, method, xtrue,
                         n_iterations, use_guessed, method_hyper_parameters)


class RunOptimiserOnRosenbrockError(RunOptimiserOnProblem):
    """
    Tests a given Optimiser on a Rosenbrock error, starting from a randomly
    sampled point with error 10.

    For constructor arguments, see :class:`RunOptimiserOnProblem`.
    """

    def __init__(self, method, n_iterations, use_guessed=False,
                 method_hyper_parameters=None):

        # Choose starting point
        c = 10
        x = np.random.uniform(-1, 3)
        y = np.sqrt((c - (1 - x)**2) / 100) + x**2
        x0 = np.array([x, y])
        sigma0 = 0.1

        # Create error
        e = pints.toy.RosenbrockError()
        x = e.optimum()
        super().__init__(e, x0, sigma0, None, None, method, x, n_iterations,
                         use_guessed, method_hyper_parameters)


class RunOptimiserOnTwoDimParabola(RunOptimiserOnProblem):
    """
    Tests a given Optimiser on a two-dimensional parabola with mean ``[0, 0]``,
    starting at a randomly chosen point 10 distance units away.

    For constructor arguments, see :class:`RunOptimiserOnProblem`.
    """

    def __init__(self, method, n_iterations, use_guessed=False,
                 method_hyper_parameters=None):
        x = np.array([0, 0])
        e = pints.toy.ParabolicError(x)
        t = np.random.uniform(0, 2 * np.pi)
        x0 = 10 * np.array([np.cos(t), np.sin(t)])
        sigma0 = 1
        super().__init__(e, x0, sigma0, None, None, method, x, n_iterations,
                         use_guessed, method_hyper_parameters)


def run_and_throw_away_warmup(controller, n_warmup):
    """ Runs sampling then throws away warmup. """
    chains = controller.run()
    return chains[:, n_warmup:]


def set_hyperparameters_for_any_mcmc_class(controller, method,
                                           method_hyper_parameters):
    """ Sets hyperparameters for any MCMC class. """
    if method_hyper_parameters is not None:
        if issubclass(method, pints.MultiChainMCMC):
            controller.sampler().set_hyper_parameters(
                method_hyper_parameters)
        else:
            for sampler in controller.samplers():
                sampler.set_hyper_parameters(method_hyper_parameters)
