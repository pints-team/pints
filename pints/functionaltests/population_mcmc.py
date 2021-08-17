#!/usr/bin/env python3
#
# Functional tests for Population MCMC.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.functionaltests as ft


def two_dim_gaussian(n_iterations=20000, n_warmup=500):
    """
    Tests :class:`pints.PopulationMCMC`
    on a two-dimensional Gaussian distribution with true solution
    ``[0, 0]`` and returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnTwoDimGaussian`.
    """
    problem = ft.RunMcmcMethodOnTwoDimGaussian(
        _method, 1, n_iterations, n_warmup)

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def banana(n_iterations=20000, n_warmup=5000):
    """
    Tests :class:`pints.PopulationMCMC`
    on a two-dimensional "twisted Gaussian" distribution with true solution
    ``[0, 0]`` and returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnBanana`.
    """
    problem = ft.RunMcmcMethodOnBanana(
        _method, 1, n_iterations, n_warmup)
    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def multimodal_gaussian(
        n_iterations=20000, n_warmup=500, n_temperatures=None):
    """
    Tests :class:`pints.PopulationMCMC`
    on a two-dimensional multi-modal Gaussian distribution with modes at
    ``[0, 0]``, ``[5, 10]``, and ``[10, 0]``, and returns a dict with entries
    "kld" and "mean-ess".

    For details of the solved problem, see
    :class:`pints.functionaltests.RunMcmcMethodOnMultimodalGaussian`.
    """
    method_hyper_parameters = None
    if n_temperatures is not None:
        method_hyper_parameters = [n_temperatures]

    problem = ft.RunMcmcMethodOnMultimodalGaussian(
        method=_method,
        n_chains=1,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
        method_hyper_parameters=method_hyper_parameters,
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


_method = pints.PopulationMCMC
_functional_tests = [
    banana,
    multimodal_gaussian,
    two_dim_gaussian,
]
