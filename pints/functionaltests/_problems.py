#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#

import numpy as np
import pints
import pints.toy


class RunMcmcMethodOnProblem(object):

    def __init__(self, log_pdf, chains):
        self.log_pdf = log_pdf
        self.chains = chains

    def estimate_kld(self):
        """
        Estimates and returns the Kullback-Leibler divergence between the
        approximate posterior and the true posterior assuming that the
        approximated posterior has Gaussian shape.
        """
        # Pool samples from chains
        chains_x = self.chains[:, :, 0].flatten()
        chains_y = self.chains[:, :, 1].flatten()
        chains = np.vstack([chains_x, chains_y]).T

        return self.log_pdf.kl_divergence(chains)

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


class RunMcmcMethodOnTwoDimGaussian(RunMcmcMethodOnProblem):
    """
    Tests a given MCMC method on a standard 2 dimensional Gaussian
    distribution.
    """

    def __init__(self, method, n_chains, n_iterations, n_warmup,
                 method_hyper_parameters=None):
        log_pdf = pints.toy.GaussianLogPDF(mean=[0, 0], sigma=[1, 1])

        # Get initial parameters
        log_prior = pints.ComposedLogPrior(
            pints.GaussianLogPrior(mean=0, sd=100),
            pints.GaussianLogPrior(mean=0, sd=100))
        initial_parameters = log_prior.sample(n=n_chains)

        # Set up sampler
        controller = pints.MCMCController(
            log_pdf, n_chains, initial_parameters, method=method)
        controller.set_max_iterations(n_iterations)
        controller.set_log_to_screen(False)
        set_hyperparameters_for_any_mcmc_class(controller, method,
                                               method_hyper_parameters)

        chains = run_and_throw_away_warmup(controller, n_warmup)

        super().__init__(log_pdf, chains)


class RunMcmcMethodOnBanana(RunMcmcMethodOnProblem):
    """
    Tests a given MCMC method on `pints.TwistedGaussianLogPDF`.
    """

    def __init__(self, method, n_chains, n_iterations, n_warmup,
                 method_hyper_parameters=None):
        log_pdf = pints.toy.TwistedGaussianLogPDF(dimension=2, b=0.1)

        # Get initial parameters
        log_prior = pints.MultivariateGaussianLogPrior([0, 0],
                                                       [[10, 0], [0, 10]])
        x0 = log_prior.sample(n_chains)
        sigma0 = np.diag(np.array([1, 3]))

        # Set up sampler
        controller = pints.MCMCController(
            log_pdf, n_chains, x0, sigma0=sigma0, method=method)
        controller.set_max_iterations(n_iterations)
        controller.set_log_to_screen(False)
        set_hyperparameters_for_any_mcmc_class(controller, method,
                                               method_hyper_parameters)

        chains = run_and_throw_away_warmup(controller, n_warmup)

        super().__init__(log_pdf, chains)


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


def run_and_throw_away_warmup(controller, n_warmup):
    """ Runs sampling then throws away warmup. """
    chains = controller.run()
    return chains[:, n_warmup:]
