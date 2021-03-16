#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#

import numpy as np
import pints
import pints.toy


class RunMcmcMethodOnProblem(object):

    def __init__(self, pdf, chains):
        self.pdf = pdf
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

        return self.pdf.kl_divergence(chains)

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

    def __init__(self, method, n_chains, n_iterations, n_warmup, method_hyper_parameters=None):
        pdf = pints.toy.GaussianLogPDF(mean=[0, 0], sigma=[1, 1])

        # Get initial parameters
        log_prior = pints.ComposedLogPrior(
            pints.GaussianLogPrior(mean=0, sd=100),
            pints.GaussianLogPrior(mean=0, sd=100))
        initial_parameters = log_prior.sample(n=n_chains)

        # Set up sampler
        controller = pints.MCMCController(
            pdf, n_chains, initial_parameters, method=method)
        controller.set_max_iterations(n_iterations)
        controller.set_log_to_screen(False)

        # Set hyper parameters, if required. This is different based on single/multi chain
        if method_hyper_parameters is not None:
            if issubclass(method, pints.MultiChainMCMC):
                controller.sampler().set_hyper_parameters(method_hyper_parameters)
            else:
                for sampler in controller.samplers():
                    sampler.set_hyper_parameters(method_hyper_parameters)

        # Infer posterior and throw away warm-up
        chains = controller.run()
        chains = chains[:, n_warmup:]

        super().__init__(pdf, chains)
