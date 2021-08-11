#!/usr/bin/env python3
#
# Functional tests for NoUTurnMCMC
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.functionaltests as ft


def two_dim_gaussian(n_iterations=None):
    """
    Tests :class:`pints.NoUTurnMCMC`
    on a two-dimensional Gaussian distribution with true solution
    ``[0, 0]`` and returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnTwoDimGaussian`.
    """
    if n_iterations is None:
        n_iterations = 1000
    problem = ft.RunMcmcMethodOnTwoDimGaussian(
        method=pints.NoUTurnMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=200
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def banana(n_iterations=None):
    """
    Tests :class:`pints.NoUTurnMCMC`
    on a two-dimensional "twisted Gaussian" distribution with true solution
    ``[0, 0]`` and returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnBanana`.
    """
    if n_iterations is None:
        n_iterations = 2000
    problem = ft.RunMcmcMethodOnBanana(
        method=pints.NoUTurnMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=500
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def high_dim_gaussian(n_iterations=None):
    """
     Tests :class:`pints.NoUTurnMCMC`
    on a 20-dimensional Gaussian distribution centered at the origin, and
    returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnHighDimensionalGaussian`.
    """
    if n_iterations is None:
        n_iterations = 4000
    problem = ft.RunMcmcMethodOnHighDimensionalGaussian(
        method=pints.NoUTurnMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=1000
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }
