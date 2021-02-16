#!/usr/bin/env python3
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#

from __future__ import division

import numpy as np

import pints
import pints.toy


class TestHaarioBardenetACMCOn2dimGaussianDistribution(object):
    """
    Tests the pints.HaarioBardenetACMC on a standard 2 dimensional Gaussian
    distribution.
    """
    def __init__(self):
        # Define calibrated test parameters
        n_chains = 3
        n_iterations = 4000
        warmup = 1000
        method = pints.HaarioBardenetACMC

        # Define pdf
        self.pdf = pints.toy.GaussianLogPDF(mean=[0, 0], sigma=[1, 1])

        # Get initial parameters
        log_prior = pints.ComposedLogPrior(
            pints.GaussianLogPrior(mean=0, sd=100),
            pints.GaussianLogPrior(mean=0, sd=100))
        initial_parameters = log_prior.sample(n=n_chains)

        # Set up sampler
        sampler = pints.MCMCController(
            self.pdf, n_chains, initial_parameters, method=method)
        sampler.set_max_iterations(n_iterations)
        sampler.set_log_to_screen(False)

        # Infer posterior and throw away warm-up
        chains = sampler.run()
        self.chains = chains[:, warmup:]

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
        # Estiomate mean ESS for each chain
        n_chains, _, n_parameters = self.chains.shape
        ess = np.empty(shape=(n_chains, n_parameters))
        for chain_id, chain in enumerate(self.chains):
            ess[chain_id] = pints.effective_sample_size(chain)

        return np.mean(ess)

    def get_results(self):
        """
        Runs the functional tests and returns the results.
        """
        results = {}

        # Estimate
        results['kld'] = self.estimate_kld()
        results['mean-ess'] = self.estimate_mean_ess()

        return results
