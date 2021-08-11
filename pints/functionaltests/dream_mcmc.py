#!/usr/bin/env python3
#
# Functional tests for DreamMCMC
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.functionaltests as ft


def two_dim_gaussian(n_iterations=10000):
    """
    Tests :class:`pints.DreamMCMC`
    on a two-dimensional Gaussian distribution with true solution
    ``[0, 0]`` and returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnTwoDimGaussian`.
    """
    n_warmup = 1000
    if n_warmup > n_iterations // 2:
        n_warmup = n_iterations // 10

    problem = ft.RunMcmcMethodOnTwoDimGaussian(
        method=pints.DreamMCMC,
        n_chains=10,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def banana(n_iterations=5000):
    """
    Tests :class:`pints.DreamMCMC`
    on a two-dimensional "twisted Gaussian" distribution with true solution
    ``[0, 0]`` and returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnBanana`.
    """
    n_warmup = 1000
    if n_warmup > n_iterations // 2:
        n_warmup = n_iterations // 10

    problem = ft.RunMcmcMethodOnBanana(
        method=pints.DreamMCMC,
        n_chains=20,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def correlated_gaussian(n_iterations=10000):
    """
    Tests :class:`pints.DreamMCMC`
    on a six-dimensional highly correlated Gaussian distribution with true
    solution ``[0, 0, 0, 0, 0, 0]`` and returns a dictionary with entries
    ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnCorrelatedGaussian`.
    """
    n_warmup = 1000
    if n_warmup > n_iterations // 2:
        n_warmup = n_iterations // 10

    problem = ft.RunMcmcMethodOnCorrelatedGaussian(
        method=pints.DreamMCMC,
        n_chains=20,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def annulus(n_iterations=10000):
    """
    Tests :class:`pints.DreamMCMC`
    on a two-dimensional annulus distribution with radius 10, and returns a
    dictionary with entries ``distance`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnAnnulus`.
    """
    n_warmup = 1000
    if n_warmup > n_iterations // 2:
        n_warmup = n_iterations // 10

    problem = ft.RunMcmcMethodOnAnnulus(
        method=pints.DreamMCMC,
        n_chains=10,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
    )

    return {
        'distance': problem.estimate_distance(),
        'mean-ess': problem.estimate_mean_ess()
    }
