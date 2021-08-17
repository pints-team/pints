#!/usr/bin/env python3
#
# Change point tests for NoUTurnMCMC
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.cptests as ft


def two_dim_gaussian(n_iterations=1000, n_warmup=200):
    """
    Tests :class:`pints.NoUTurnMCMC`
    on a two-dimensional Gaussian distribution with true solution
    ``[0, 0]`` and returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.cptests.RunMcmcMethodOnTwoDimGaussian`.
    """
    problem = ft.RunMcmcMethodOnTwoDimGaussian(
        _method, 4, n_iterations, n_warmup)
    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def banana(n_iterations=2000, n_warmup=500):
    """
    Tests :class:`pints.NoUTurnMCMC`
    on a two-dimensional "twisted Gaussian" distribution with true solution
    ``[0, 0]`` and returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.cptests.RunMcmcMethodOnBanana`.
    """
    problem = ft.RunMcmcMethodOnBanana(
        _method, 4, n_iterations, n_warmup)
    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def high_dim_gaussian(n_iterations=4000, n_warmup=1000):
    """
     Tests :class:`pints.NoUTurnMCMC`
    on a 20-dimensional Gaussian distribution centered at the origin, and
    returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.cptests.RunMcmcMethodOnHighDimensionalGaussian`.
    """
    problem = ft.RunMcmcMethodOnHighDimensionalGaussian(
        _method, 4, n_iterations, n_warmup)
    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


_method = pints.NoUTurnMCMC
_change_point_tests = [
    banana,
    high_dim_gaussian,
    two_dim_gaussian,
]
