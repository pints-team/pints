#!/usr/bin/env python3
#
# Functional tests for MonomialGammaHamiltonianMCMC
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.functionaltests as ft


def two_dim_gaussian(n_iterations=None):
    """
    Tests :class:`pints.MonomialGammaHamiltonianMCMC`
    on a two-dimensional Gaussian distribution with true solution
    ``[0, 0]`` and returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnTwoDimGaussian`.
    """
    if n_iterations is None:
        n_iterations = 1000
    problem = ft.RunMcmcMethodOnTwoDimGaussian(
        method=pints.MonomialGammaHamiltonianMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=200,
        method_hyper_parameters=[20, 1, 0.5, 0.2, 1]
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def banana(n_iterations=None):
    """
    Tests :class:`pints.MonomialGammaHamiltonianMCMC`
    on a two-dimensional "twisted Gaussian" distribution with true solution
    ``[0, 0]`` and returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnBanana`.
    """
    if n_iterations is None:
        n_iterations = 2000
    problem = ft.RunMcmcMethodOnBanana(
        method=pints.MonomialGammaHamiltonianMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=500,
        method_hyper_parameters=[20, 1, 0.5, 0.2, 1]
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def high_dim_gaussian(n_iterations=None):
    """
     Tests :class:`pints.MonomialGammaHamiltonianMCMC`
    on a 20-dimensional Gaussian distribution centered at the origin, and
    returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnHighDimensionalGaussian`.
    """
    if n_iterations is None:
        n_iterations = 2000
    problem = ft.RunMcmcMethodOnHighDimensionalGaussian(
        method=pints.MonomialGammaHamiltonianMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=500,
        method_hyper_parameters=[20, 1, 0.5, 0.2, 1]
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def correlated_gaussian(n_iterations=None):
    """
    Tests :class:`pints.MonomialGammaHamiltonianMCMC`
    on a six-dimensional highly correlated Gaussian distribution with true
    solution ``[0, 0, 0, 0, 0, 0]`` and returns a dictionary with entries
    ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnCorrelatedGaussian`.
    """
    if n_iterations is None:
        n_iterations = 2000
    problem = ft.RunMcmcMethodOnCorrelatedGaussian(
        method=pints.MonomialGammaHamiltonianMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=500,
        method_hyper_parameters=[20, 1, 0.5, 0.2, 1]
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def annulus(n_iterations=None):
    """
    Tests :class:`pints.MonomialGammaHamiltonianMCMC`
    on a two-dimensional annulus distribution with radius 10, and returns a
    dictionary with entries ``distance`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnAnnulus`.
    """
    if n_iterations is None:
        n_iterations = 2000
    problem = ft.RunMcmcMethodOnAnnulus(
        method=pints.MonomialGammaHamiltonianMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=500,
        method_hyper_parameters=[20, 1, 0.5, 0.2, 1]
    )

    return {
        'distance': problem.estimate_distance(),
        'mean-ess': problem.estimate_mean_ess()
    }


def multimodal_gaussian(n_iterations=None):
    """
    Tests :class:`pints.MonomialGammaHamiltonianMCMC`
    on a two-dimensional multi-modal Gaussian distribution with modes at
    ``[0, 0]``, ``[5, 10]``, and ``[10, 0]``, and returns a dict with entries
    "kld" and "mean-ess".

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnMultimodalGaussian`.
    """
    if n_iterations is None:
        n_iterations = 2000
    problem = ft.RunMcmcMethodOnMultimodalGaussian(
        method=pints.MonomialGammaHamiltonianMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=500,
        method_hyper_parameters=[20, 1, 0.5, 0.2, 1]
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def cone(n_iterations=None):
    """
    Tests :class:`pints.MonomialGammaHamiltonianMCMC`
    on a two-dimensional cone distribution centered at ``[0, 0]``, and returns
    a dict with entries "distance" and "mean-ess".

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnCone`.
    """
    if n_iterations is None:
        n_iterations = 2000
    problem = ft.RunMcmcMethodOnCone(
        method=pints.MonomialGammaHamiltonianMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=500,
        method_hyper_parameters=[20, 1, 0.5, 0.2, 1]
    )

    return {
        'distance': problem.estimate_distance(),
        'mean-ess': problem.estimate_mean_ess()
    }
