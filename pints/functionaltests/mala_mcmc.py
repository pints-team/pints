#!/usr/bin/env python3
#
# Functional tests for MALA MCMC
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.functionaltests as ft


def two_dim_gaussian(n_iterations=1000):
    """
    Tests :class:`pints.MALAMCMC`
    on a two-dimensional Gaussian distribution with true solution
    ``[0, 0]`` and returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnTwoDimGaussian`.
    """
    n_warmup = 200
    if n_warmup > n_iterations // 2:
        n_warmup = n_iterations // 10

    problem = ft.RunMcmcMethodOnTwoDimGaussian(
        method=pints.MALAMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
        method_hyper_parameters=[[1.0, 1.0]]
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def banana(n_iterations=2000):
    """
    Tests :class:`pints.MALAMCMC`
    on a two-dimensional "twisted Gaussian" distribution with true solution
    ``[0, 0]`` and returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnBanana`.
    """
    n_warmup = 500
    if n_warmup > n_iterations // 2:
        n_warmup = n_iterations // 10

    problem = ft.RunMcmcMethodOnBanana(
        method=pints.MALAMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
        method_hyper_parameters=[[0.8] * 2]
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def high_dim_gaussian(n_iterations=2000):
    """
     Tests :class:`pints.MALAMCMC`
    on a 20-dimensional Gaussian distribution centered at the origin, and
    returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnHighDimensionalGaussian`.
    """
    n_warmup = 500
    if n_warmup > n_iterations // 2:
        n_warmup = n_iterations // 10

    problem = ft.RunMcmcMethodOnHighDimensionalGaussian(
        method=pints.MALAMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
        method_hyper_parameters=[[1.2] * 20]
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def correlated_gaussian(n_iterations=2000):
    """
    Tests :class:`pints.MALAMCMC`
    on a six-dimensional highly correlated Gaussian distribution with true
    solution ``[0, 0, 0, 0, 0, 0]`` and returns a dictionary with entries
    ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnCorrelatedGaussian`.
    """
    n_warmup = 500
    if n_warmup > n_iterations // 2:
        n_warmup = n_iterations // 10

    problem = ft.RunMcmcMethodOnCorrelatedGaussian(
        method=pints.MALAMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
        method_hyper_parameters=[[1.0] * 6]
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def annulus(n_iterations=2000):
    """
    Tests :class:`pints.MALAMCMC`
    on a two-dimensional annulus distribution with radius 10, and returns a
    dictionary with entries ``distance`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnAnnulus`.
    """
    n_warmup = 500
    if n_warmup > n_iterations // 2:
        n_warmup = n_iterations // 10

    problem = ft.RunMcmcMethodOnAnnulus(
        method=pints.MALAMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
        method_hyper_parameters=[[1.2] * 2],
    )

    return {
        'distance': problem.estimate_distance(),
        'mean-ess': problem.estimate_mean_ess()
    }


def multimodal_gaussian(n_iterations=2000):
    """
    Tests :class:`pints.MALAMCMC`
    on a two-dimensional multi-modal Gaussian distribution with modes at
    ``[0, 0]``, ``[5, 10]``, and ``[10, 0]``, and returns a dict with entries
    "kld" and "mean-ess".

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnMultimodalGaussian`.
    """
    n_warmup = 500
    if n_warmup > n_iterations // 2:
        n_warmup = n_iterations // 10

    problem = ft.RunMcmcMethodOnMultimodalGaussian(
        method=pints.MALAMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
        method_hyper_parameters=[[2.0] * 2]
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def cone(n_iterations=2000):
    """
    Tests :class:`pints.MALAMCMC`
    on a two-dimensional cone distribution centered at ``[0, 0]``, and returns
    a dict with entries "distance" and "mean-ess".

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnCone`.
    """
    n_warmup = 500
    if n_warmup > n_iterations // 2:
        n_warmup = n_iterations // 10

    problem = ft.RunMcmcMethodOnCone(
        method=pints.MALAMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
        method_hyper_parameters=[[1.0, 1.0]],
    )

    return {
        'distance': problem.estimate_distance(),
        'mean-ess': problem.estimate_mean_ess()
    }


_method = pints.MALAMCMC
_functional_tests = [
    annulus,
    banana,
    cone,
    correlated_gaussian,
    high_dim_gaussian,
    multimodal_gaussian,
    two_dim_gaussian,
]
