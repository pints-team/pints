#!/usr/bin/env python3
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.functionaltests as ft


def two_dim_gaussian(n_iterations=None):
    if n_iterations is None:
        n_iterations = 5000
    problem = ft.RunMcmcMethodOnTwoDimGaussian(
        method=pints.SliceStepoutMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=500
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }

def correlated_gaussian(n_iterations=None):
    if n_iterations is None:
        n_iterations = 5000
    problem = ft.RunMcmcMethodOnCorrelatedGaussian(
        method=pints.SliceStepoutMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=500
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def banana(n_iterations=None):
    if n_iterations is None:
        n_iterations = 5000
    problem = ft.RunMcmcMethodOnBanana(
        method=pints.SliceStepoutMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=500
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def high_dim_gaussian(n_iterations=None):
    if n_iterations is None:
        n_iterations = 5000
    problem = ft.RunMcmcMethodOnHighDimensionalGaussian(
        method=pints.SliceStepoutMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=500
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def annulus(n_iterations=None):
    if n_iterations is None:
        n_iterations = 10000
    problem = ft.RunMcmcMethodOnAnnulus(
        method=pints.SliceStepoutMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=2000
    )

    return {
        'distance': problem.estimate_distance(),
        'mean-ess': problem.estimate_mean_ess()
    }


def multimodal_gaussian(n_iterations=None):
    if n_iterations is None:
        n_iterations = 5000
    problem = ft.RunMcmcMethodOnMultimodalGaussian(
        method=pints.SliceStepoutMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=500
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def cone(n_iterations=None):
    if n_iterations is None:
        n_iterations = 5000
    problem = ft.RunMcmcMethodOnCone(
        method=pints.SliceStepoutMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=500
    )

    return {
        'distance': problem.estimate_distance(),
        'mean-ess': problem.estimate_mean_ess()
    }
