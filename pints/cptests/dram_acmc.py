#!/usr/bin/env python3
#
# Change point tests for DramACMC
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.cptests as cpt


def two_dim_gaussian(n_iterations=8000, n_warmup=2000):
    """
    Tests :class:`pints.DramACMC`
    on a two-dimensional Gaussian distribution with true solution
    ``[0, 0]`` and returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.cptests.RunMcmcMethodOnTwoDimGaussian`.
    """
    problem = cpt.RunMcmcMethodOnTwoDimGaussian(
        _method, 4, n_iterations, n_warmup)
    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def banana(n_iterations=4000, n_warmup=1000):
    """
    Tests :class:`pints.DramACMC`
    on a two-dimensional "twisted Gaussian" distribution with true solution
    ``[0, 0]`` and returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.cptests.RunMcmcMethodOnBanana`.
    """
    problem = cpt.RunMcmcMethodOnBanana(
        _method, 4, n_iterations, n_warmup)
    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def correlated_gaussian(n_iterations=8000, n_warmup=4000):
    """
    Tests :class:`pints.DramACMC`
    on a six-dimensional highly correlated Gaussian distribution with true
    solution ``[0, 0, 0, 0, 0, 0]`` and returns a dictionary with entries
    ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.cptests.RunMcmcMethodOnCorrelatedGaussian`.
    """
    problem = cpt.RunMcmcMethodOnCorrelatedGaussian(
        _method, 4, n_iterations, n_warmup)
    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


_method = pints.DramACMC
_change_point_tests = [
    banana,
    correlated_gaussian,
    two_dim_gaussian,
]
