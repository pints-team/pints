#!/usr/bin/env python3
#
# Change point tests for SliceDoublingMCMC
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.cptests as cpt


def two_dim_gaussian(n_iterations=5000, n_warmup=500):
    """
    Tests :class:`pints.SliceDoublingMCMC`
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


def correlated_gaussian(n_iterations=5000, n_warmup=500):
    """
    Tests :class:`pints.SliceDoublingMCMC`
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


def high_dim_gaussian(n_iterations=5000, n_warmup=500):
    """
     Tests :class:`pints.SliceDoublingMCMC`
    on a 20-dimensional Gaussian distribution centered at the origin, and
    returns a dictionary with entries ``kld`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.cptests.RunMcmcMethodOnHighDimensionalGaussian`.
    """
    problem = cpt.RunMcmcMethodOnHighDimensionalGaussian(
        _method, 4, n_iterations, n_warmup)
    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def annulus(n_iterations=10000, n_warmup=2000):
    """
    Tests :class:`pints.SliceDoublingMCMC`
    on a two-dimensional annulus distribution with radius 10, and returns a
    dictionary with entries ``distance`` and ``mean-ess``.

    For details of the solved problem, see
    :class:`pints.cptests.RunMcmcMethodOnAnnulus`.
    """
    problem = cpt.RunMcmcMethodOnAnnulus(
        _method, 4, n_iterations, n_warmup)
    return {
        'distance': problem.estimate_distance(),
        'mean-ess': problem.estimate_mean_ess()
    }


def cone(n_iterations=5000, n_warmup=500):
    """
    Tests :class:`pints.SliceDoublingMCMC`
    on a two-dimensional cone distribution centered at ``[0, 0]``, and returns
    a dict with entries "distance" and "mean-ess".

    For details of the solved problem, see
    :class:`pints.cptests.RunMcmcMethodOnCone`.
    """
    problem = cpt.RunMcmcMethodOnCone(
        _method, 4, n_iterations, n_warmup)
    return {
        'distance': problem.estimate_distance(),
        'mean-ess': problem.estimate_mean_ess()
    }


_method = pints.SliceStepoutMCMC
_change_point_tests = [
    annulus,
    cone,
    correlated_gaussian,
    high_dim_gaussian,
    two_dim_gaussian,
]
