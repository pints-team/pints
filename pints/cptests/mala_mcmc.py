#!/usr/bin/env python3
#
# Functional tests for MALA MCMC
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.cptests as ft


def two_dim_gaussian(n_iterations=1000, n_warmup=200):
    """
    Tests :class:`pints.MALAMCMC`
    on a two-dimensional Gaussian distribution with true solution
    ``[0, 0]`` and returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.cptests.RunMcmcMethodOnTwoDimGaussian`.
    """
    problem = ft.RunMcmcMethodOnTwoDimGaussian(
        method=_method,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
        method_hyper_parameters=[[1.0, 1.0]]
    )
    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def banana(n_iterations=2000, n_warmup=500):
    """
    Tests :class:`pints.MALAMCMC`
    on a two-dimensional "twisted Gaussian" distribution with true solution
    ``[0, 0]`` and returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.cptests.RunMcmcMethodOnBanana`.
    """
    problem = ft.RunMcmcMethodOnBanana(
        method=_method,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
        method_hyper_parameters=[[0.8] * 2]
    )
    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def high_dim_gaussian(n_iterations=2000, n_warmup=500):
    """
     Tests :class:`pints.MALAMCMC`
    on a 20-dimensional Gaussian distribution centered at the origin, and
    returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.cptests.RunMcmcMethodOnHighDimensionalGaussian`.
    """
    problem = ft.RunMcmcMethodOnHighDimensionalGaussian(
        method=_method,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
        method_hyper_parameters=[[1.2] * 20]
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def correlated_gaussian(n_iterations=2000, n_warmup=500):
    """
    Tests :class:`pints.MALAMCMC`
    on a six-dimensional highly correlated Gaussian distribution with true
    solution ``[0, 0, 0, 0, 0, 0]`` and returns a dictionary with entries
    ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.cptests.RunMcmcMethodOnCorrelatedGaussian`.
    """
    problem = ft.RunMcmcMethodOnCorrelatedGaussian(
        method=_method,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
        method_hyper_parameters=[[1.0] * 6]
    )
    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def annulus(n_iterations=2000, n_warmup=500):
    """
    Tests :class:`pints.MALAMCMC`
    on a two-dimensional annulus distribution with radius 10, and returns a
    dictionary with entries ``distance`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.cptests.RunMcmcMethodOnAnnulus`.
    """
    problem = ft.RunMcmcMethodOnAnnulus(
        method=_method,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
        method_hyper_parameters=[[1.2] * 2],
    )
    return {
        'distance': problem.estimate_distance(),
        'mean-ess': problem.estimate_mean_ess()
    }


def multimodal_gaussian(n_iterations=2000, n_warmup=500):
    """
    Tests :class:`pints.MALAMCMC`
    on a two-dimensional multi-modal Gaussian distribution with modes at
    ``[0, 0]``, ``[5, 10]``, and ``[10, 0]``, and returns a dict with entries
    "kld" and "mean-ess".

    For details of the solved problem, see
    :class:`pints.cptests.RunMcmcMethodOnMultimodalGaussian`.
    """
    problem = ft.RunMcmcMethodOnMultimodalGaussian(
        method=_method,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
        method_hyper_parameters=[[2.0] * 2]
    )
    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def cone(n_iterations=2000, n_warmup=500):
    """
    Tests :class:`pints.MALAMCMC`
    on a two-dimensional cone distribution centered at ``[0, 0]``, and returns
    a dict with entries "distance" and "mean-ess".

    For details of the solved problem, see
    :class:`pints.cptests.RunMcmcMethodOnCone`.
    """
    problem = ft.RunMcmcMethodOnCone(
        method=_method,
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
_change_point_tests = [
    annulus,
    banana,
    cone,
    correlated_gaussian,
    high_dim_gaussian,
    multimodal_gaussian,
    two_dim_gaussian,
]
