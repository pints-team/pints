#
# Uses the Python `pymc3` (and `theano`) module to run SMC routine sampling.
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import numpy as np
import theano
import theano.tensor as tt
from theano.compile.ops import as_op
import pymc3 as pm
# from pymc3.step_methods import smc


# Set some flags for theano
THEANO_FLAGS = 'floatX=float32,optimizer=fast_compile'
theano.config.floatX = 'float64'
theano.config.compute_test_value = 'raise'
theano.config.exception_verbosity = 'high'


#TODO: issue due to @as_op cannot be done inside class
global evaluate, times


class SMC(pints.MCMC):
    """
    *Extends:* :class:`MCMC`

    Creates a chain of samples from a target distribution, using the SMC [1]
    routine implemented in the `pymc3` module [2] with `theano` backend [3].

    SMC stands for Sequential Monte Carlo.

    [1] Ching, Chen, "Transitional Markov Chain Monte Carlo method for Bayesian
    model updating, model class selection, and model averaging", Journal of
    engineering mechanics, vol. 133, no. 7, p. 816-832, 2007.

    [2] Patil, Huard, Fonnesbeck, "PyMC: Bayesian stochastic modelling in
    Python", Journl of statistical software, vol. 35, no. 4, p. 1, 2010.

    [3] Bergstra, Breuleux, Bastien, Lamblin, Pascanu, Desjardins, Turian,
    Warde-Farley, Bengio, "Theano: A CPU and GPU Math compiler in Python", in
    Proc. 9th Python in Science Conf., p. 1-7, 2010.

    """
    def __init__(self, likelihood, x0, sigma0=None):
        super(SMC, self).__init__(likelihood, x0, sigma0)
        self._problem = likelihood.problem()
        self._evaluate = self._problem.evaluate

        # pass the following to the _tt_evaluate functions
        # TODO: @as_op cannot be done inside class
        global evaluate, times
        evaluate = self._evaluate
        times = self._problem.times()

        # Tensorflow wrapper
        if isinstance(self._problem, pints.SingleSeriesProblem):
            # For single series problem
            self._tt_evaluate = _tt_evaluate_single_series
            self._values = np.array([self._problem.values()]).T
        else:
            # For multiple series problem
            raise NotImplementedError

        # Check problem assumption:
        # Guassian log_likelihood and should have prior
        if not isinstance(self._log_likelihood, pints.BayesianLogLikelihood):
            raise NotImplementedError(
                'Currently only support pints.BayesianLogLikelihood')

        # Marginal Likelihood
        self._marginal_likelihood = None

    def _get_prior_as_pm(self, prior):
        """
        Return PyMC3 type prior.
        """
        if isinstance(prior, pints.UniformPrior):
            boundaries = prior._boundaries
            # Assuming the last one is sigma
            prior_pm = pm.Uniform(
                'prior',
                lower=boundaries.lower()[:-1],
                upper=boundaries.upper()[:-1],
                transform=None,
                shape=self._dimension - 1)
            return prior_pm
        elif isinstance(prior, pints.NormalPrior):
            mean = prior._mean
            cov = -0.5 / prior._inv2cov
            # TODO: Cannot be univariate...(?)
            raise NotImplementedError('Provided Prior not yet implemented.')
        elif isinstance(prior, pints.MultivariateNormalPrior):
            mean = prior._mean
            cov = prior._cov
            # Assuming the last one is sigma (and independent from the others)
            prior_pm = pm.MvNormal(
                'prior',
                mu=mean[:-1],
                cov=cov[:-1, :-1],
                shape=self._dimension - 1)
            return prior_pm
        elif isinstance(prior, pints.Prior):
            raise NotImplementedError('Provided Prior not yet implemented.')
        else:
            raise NotImplementedError('pints.Prior object is expected.')

    def _get_sigma_as_pm(self, prior):
        """
        Return PyMC3 type prior.
        """
        if isinstance(prior, pints.UniformPrior):
            boundaries = prior._boundaries
            # Assuming the last one is sigma
            prior_pm = pm.Uniform(
                'sigma',
                lower=boundaries.lower()[-1],
                upper=boundaries.upper()[-1],
                transform=None)
            return prior_pm
        elif isinstance(prior, pints.NormalPrior):
            mean = prior._mean
            cov = -0.5 / prior._inv2cov
            # TODO: Cannot be univariate...(?)
            raise NotImplementedError('Provided Prior not yet implemented.')
        elif isinstance(prior, pints.MultivariateNormalPrior):
            mean = prior._mean
            cov = prior._cov
            # Assuming the last one is sigma (and independent from the others)
            prior_pm = pm.Normal(
                'sigma',
                mu=mean[-1],
                sd=cov[-1, -1])
            return prior_pm
        elif isinstance(prior, pints.Prior):
            raise NotImplementedError('Provided Prior not yet implemented.')
        else:
            raise NotImplementedError('pints.Prior object is expected.')

    def tt_evaluate(self, x):
        """
        Return Theano tensor object.
        """
        return self._tt_evaluate(x)

    def run(self):
        from tempfile import mkdtemp

        # TODO: allow changes before run
        # number of chains
        n_chains = 100

        # number of steps
        n_steps = 20

        # tuning interval
        tune_interval = 5

        # number of jobs
        n_jobs = 10

        # starting stage
        stage = 0

        # initial spread
        sigma_x0 = 0.2  # float or array_like of floats

        # output folder
        homepath = mkdtemp(prefix='PINTS_SMC')

        # random seed
        seed = 21

        # Starting point
        x0 = self._x0

        # Start SMC setup
        pm_model = pm.Model()
        with pm_model:
            # setup prior as PyMC3
            # pm_prior = self._get_prior_as_pm(self._log_likelihood._prior)
            # pm_sigma = self._get_sigma_as_pm(self._log_likelihood._prior)

            # setup model as PyMC3
            # Assuming we are doing pints.BayesianLogLikelihood
            # mu = self._tt_evaluate(pm_prior)
            # cov = np.eye(1) * pm_sigma**2

            # setup likelihood as PyMC3
            # Assuming Guassian
            # pm_likelihood = pm.MvNormal('likelihood', mu=mu, cov=cov,
            #                            observed=self._values)

            # Guess x0
            start = [
                {'prior': np.random.normal(x0[:-1], x0[:-1] * sigma_x0),
                 'sigma': np.random.normal(x0[-1], x0[-1] * sigma_x0)}
                for _ in range(n_chains)]

            # run!
            pm_chain = pm.step_methods.smc.sample_smc(
                n_steps=n_steps,
                n_chains=n_chains,
                progressbar=self._verbose,
                homepath=homepath,
                n_jobs=n_jobs,
                stage=stage,
                start=start,
                tune_interval=tune_interval,
                random_seed=seed
            )

        # Get marginal likelihood
        self._marginal_likelihood = pm_model.marginal_likelihood

        # Convert it to numpy
        from pymc3.plots.utils import get_default_varnames
        varname = get_default_varnames(pm_chain.varnames, False)
        shape = pm_chain.get_values(
            varname[0], combine=True, squeeze=True).shape
        chain = np.zeros((shape[0], shape[1] + 1))
        chain[:, :-1] = pm_chain.get_values(
            varname[0], combine=True, squeeze=True)
        chain[:, -1] = pm_chain.get_values(
            varname[1], combine=True, squeeze=True)

        # Return generated chain
        return chain

    def marginal_likelihood(self):
        """
        Return the computed marginal likelihood through the SMC routine.

        Return None if SMC.run() not yet run.
        """
        return self._marginal_likelihood


@as_op(itypes=[tt.dvector], otypes=[tt.dmatrix])
def _tt_evaluate_single_series(x):
    sol = evaluate(x)
    try:
        if not np.isfinite(sol):
            # If the function return some weird things
            return np.array([[1e10] * len(times)]).T
    except Exception:
        return np.array([sol]).T


@as_op(itypes=[tt.dvector], otypes=[tt.dmatrix])
def _tt_evaluate_multiple_series(x):
    sol = evaluate(x)
    return np.array([sol]).T


def smc(log_likelihood, x0, sigma0=None):
    """
    Runs a SMC routine with the default parameters.
    """
    return SMC(log_likelihood, x0, sigma0).run()

