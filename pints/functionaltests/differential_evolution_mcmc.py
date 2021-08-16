#!/usr/bin/env python3
#
# Functional tests for DifferentialEvolutionMCMC
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.functionaltests as ft


def two_dim_gaussian(n_iterations=10000, n_warmup=1000):
    """
    Tests :class:`pints.DifferentialEvolutionMCMC`
    on a two-dimensional Gaussian distribution with true solution
    ``[0, 0]`` and returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnTwoDimGaussian`.
    """
    problem = ft.RunMcmcMethodOnTwoDimGaussian(
        method=pints.DifferentialEvolutionMCMC,
        n_chains=10,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def banana(n_iterations=5000, n_warmup=1000):
    """
    Tests :class:`pints.DifferentialEvolutionMCMC`
    on a two-dimensional "twisted Gaussian" distribution with true solution
    ``[0, 0]`` and returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnBanana`.
    """
    problem = ft.RunMcmcMethodOnBanana(
        method=pints.DifferentialEvolutionMCMC,
        n_chains=20,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def correlated_gaussian(n_iterations=10000, n_warmup=1000):
    """
    Tests :class:`pints.DifferentialEvolutionMCMC`
    on a six-dimensional highly correlated Gaussian distribution with true
    solution ``[0, 0, 0, 0, 0, 0]`` and returns a dictionary with entries
    ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnCorrelatedGaussian`.
    """
    problem = ft.RunMcmcMethodOnCorrelatedGaussian(
        method=pints.DifferentialEvolutionMCMC,
        n_chains=20,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def annulus(n_iterations=10000, n_warmup=1000):
    """
    Tests :class:`pints.DifferentialEvolutionMCMC`
    on a two-dimensional annulus distribution with radius 10, and returns a
    dictionary with entries ``distance`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnAnnulus`.
    """
    problem = ft.RunMcmcMethodOnAnnulus(
        method=pints.DifferentialEvolutionMCMC,
        n_chains=10,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
    )

    return {
        'distance': problem.estimate_distance(),
        'mean-ess': problem.estimate_mean_ess()
    }


_method = pints.DifferentialEvolutionMCMC
_functional_tests = [
    annulus,
    banana,
    correlated_gaussian,
    two_dim_gaussian,
]
