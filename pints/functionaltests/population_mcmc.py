#!/usr/bin/env python3
#
# Functional tests for Population MCMC.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints

from ._problems import (
    RunMcmcMethodOnBanana,
    RunMcmcMethodOnMultimodalGaussian,
    RunMcmcMethodOnSimpleEggBox,
    RunMcmcMethodOnTwoDimGaussian,
)


def test_population_mcmc_on_two_dim_gaussian(n_iterations=None):
    """
    Tests :class:`pints.PopulationMCMC` on a two-dimensional Gaussian with
    means 0 and 0, and returns a dict with entries "kld" and "mean-ess".

    See :class:`pints
    """
    if n_iterations is None:
        n_iterations = 20000

    n_warmup = 500
    if n_warmup > n_iterations // 2:
        n_warmup = n_iterations // 10

    problem = RunMcmcMethodOnTwoDimGaussian(
        method=pints.PopulationMCMC,
        n_chains=1,
        n_iterations=n_iterations,
        n_warmup=n_warmup
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def test_population_mcmc_on_banana(n_iterations=None):
    """
    Tests :class:`pints.PopulationMCMC` on a 2-d twisted Gaussian "banana"
    problem with true solution ``(0, 0)``, and returns a dict with entries
    "kld" and "mean-ess".
    """
    if n_iterations is None:
        n_iterations = 20000

    n_warmup = 5000  # Needs a lot of warm-up on banana!
    if n_warmup > n_iterations // 2:
        n_warmup = n_iterations // 10

    problem = RunMcmcMethodOnBanana(
        method=pints.PopulationMCMC,
        n_chains=1,
        n_iterations=n_iterations,
        n_warmup=n_warmup
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def test_population_mcmc_on_multimodal_gaussian(
        n_iterations=None, n_temperatures=None):
    """
    Tests :class:`pints.PopulationMCMC` on a multi-modal Gaussian distribution
    with modes at ``[0, 0]``, ``[5, 10]``, and ``[10, 0]``, and returns a dict
    with entries "kld" and "mean-ess".
    """
    if n_iterations is None:
        n_iterations = 20000

    n_warmup = 500
    if n_warmup > n_iterations // 2:
        n_warmup = n_iterations // 10

    method_hyper_parameters = None
    if n_temperatures is not None:
        method_hyper_parameters = [n_temperatures]

    problem = RunMcmcMethodOnMultimodalGaussian(
        method=pints.PopulationMCMC,
        n_chains=1,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
        method_hyper_parameters=method_hyper_parameters,
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }

