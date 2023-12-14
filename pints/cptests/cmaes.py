#!/usr/bin/env python3
#
# Change point tests for CMAES.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.cptests as cpt


def bounded_fitzhugh_nagumo(n_iterations=100):
    """
    Tests :class:`pints.CMAES` on a bounded Fitzhugh-Nagumo model, and returns
    a dictionary with ``error`` and ``distance``.

    For details of the solved problem, see
    :class:`pints.cptests.RunOptimiserOnBoundedUntransformedLogistic`.
    """
    problem = cpt.RunOptimiserOnBoundedFitzhughNagumo(
        _method, n_iterations, _fguess)
    return {
        'error': problem.error(),
        'distance': problem.distance()
    }


def bounded_untransformed_logistic(n_iterations=300):
    """
    Tests :class:`pints.CMAES` on a bounded logistic model without
    transformations, and returns a dictionary with ``error`` and ``distance``.

    For details of the solved problem, see
    :class:`pints.cptests.RunOptimiserOnBoundedUntransformedLogistic`.
    """
    problem = cpt.RunOptimiserOnBoundedUntransformedLogistic(
        _method, n_iterations, _fguess)
    return {
        'error': problem.error(),
        'distance': problem.distance()
    }


def rosenbrock(n_iterations=100):
    """
    Tests :class:`pints.CMAES` on a Rosenbrock error and returns a dictionary
    with ``error`` and ``distance``.

    For details of the solved problem, see
    :class:`pints.cptests.RunOptimiserOnRosenbrockError`.
    """
    problem = cpt.RunOptimiserOnRosenbrockError(_method, n_iterations, _fguess)
    return {
        'error': problem.error(),
        'distance': problem.distance()
    }


def two_dim_parabola(n_iterations=50):
    """
    Tests :class:`pints.CMAES` on a two-dimensional parabolic error and returns
    a dictionary with entries ``error`` and ``distance``.

    For details of the solved problem, see
    :class:`pints.cptests.RunOptimiserOnTwoDimParabola`.
    """
    problem = cpt.RunOptimiserOnTwoDimParabola(_method, n_iterations, _fguess)
    return {
        'error': problem.error(),
        'distance': problem.distance()
    }


_method = pints.CMAES
_fguess = True
_change_point_tests = [
    bounded_untransformed_logistic,
    rosenbrock,
    two_dim_parabola,
]
