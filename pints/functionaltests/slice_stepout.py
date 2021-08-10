#!/usr/bin/env python3
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#

from __future__ import division

import pints

from ._problems import (RunMcmcMethodOnTwoDimGaussian,
                        RunMcmcMethodOnBanana,
                        RunMcmcMethodOnHighDimensionalGaussian,
                        RunMcmcMethodOnCorrelatedGaussian,
                        RunMcmcMethodOnAnnulus,
                        RunMcmcMethodOnMultimodalGaussian,
                        RunMcmcMethodOnCone)


def test_slice_stepout_on_two_dim_gaussian(n_iterations=None):
    if n_iterations is None:
        n_iterations = 5000
    problem = RunMcmcMethodOnTwoDimGaussian(
        method=pints.SliceStepoutMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=500
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }

def test_slice_stepout_on_correlated_gaussian(n_iterations=None):
    if n_iterations is None:
        n_iterations = 5000
    problem = RunMcmcMethodOnCorrelatedGaussian(
        method=pints.SliceStepoutMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=500
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def test_slice_stepout_on_banana(n_iterations=None):
    if n_iterations is None:
        n_iterations = 5000
    problem = RunMcmcMethodOnBanana(
        method=pints.SliceStepoutMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=500
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def test_slice_stepout_on_high_dim_gaussian(n_iterations=None):
    if n_iterations is None:
        n_iterations = 5000
    problem = RunMcmcMethodOnHighDimensionalGaussian(
        method=pints.SliceStepoutMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=500
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def test_slice_stepout_on_annulus(n_iterations=None):
    if n_iterations is None:
        n_iterations = 10000
    problem = RunMcmcMethodOnAnnulus(
        method=pints.SliceStepoutMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=2000
    )

    return {
        'distance': problem.estimate_distance(),
        'mean-ess': problem.estimate_mean_ess()
    }


def test_slice_stepout_on_multimodal_gaussian(n_iterations=None):
    if n_iterations is None:
        n_iterations = 5000
    problem = RunMcmcMethodOnMultimodalGaussian(
        method=pints.SliceStepoutMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=500
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def test_slice_stepout_on_cone(n_iterations=None):
    if n_iterations is None:
        n_iterations = 5000
    problem = RunMcmcMethodOnCone(
        method=pints.SliceStepoutMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=500
    )

    return {
        'distance': problem.estimate_distance(),
        'mean-ess': problem.estimate_mean_ess()
    }
