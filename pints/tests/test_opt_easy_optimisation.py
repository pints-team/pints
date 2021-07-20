#!/usr/bin/env python3
#
# Tests the easy optimisation methods fmin and curve_fit.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import unittest
import numpy as np


class TestEasyOptimisation(unittest.TestCase):
    """
    Tests the easy optimisation methods fmin and curve_fit.
    """
    def test_fmin(self):
        # Tests :meth:`pints.fmin()`.

        # Note: This just wraps around `OptimisationController`, so testing
        # done here is for wrapper code, not main functionality!

        # Basic test
        np.random.seed(1)
        xopt, fopt = pints.fmin(f, [1, 1], method=pints.XNES)
        self.assertAlmostEqual(xopt[0], 3)
        self.assertAlmostEqual(xopt[1], -5)

        # Function must be callable
        self.assertRaisesRegex(ValueError, 'callable', pints.fmin, 3, [1])

        # Test with boundaries
        xopt, fopt = pints.fmin(
            f, [1, 1], boundaries=([-10, -10], [10, 10]), method=pints.SNES)
        self.assertAlmostEqual(xopt[0], 3)
        self.assertAlmostEqual(xopt[1], -5)

        # Test with extra arguments
        def g(x, y, z):
            return (x[0] - 3) ** 2 + (x[1] + 5) ** 2 + y / z
        xopt, fopt = pints.fmin(g, [1, 1], args=[1, 2], method=pints.XNES)
        self.assertAlmostEqual(xopt[0], 3)
        self.assertAlmostEqual(xopt[1], -5)

        # Test with parallelisation
        pints.fmin(f, [1, 1], parallel=True, method=pints.XNES)

    def test_curve_fit(self):
        # Tests :meth:`pints.curve_fit()`.

        # Note: This just wraps around `OptimisationController`, so testing
        # done here is for wrapper code, not main functionality!
        np.random.seed(1)

        # Basic test
        x = np.linspace(-5, 5, 100)
        e = np.random.normal(loc=0, scale=0.1, size=x.shape)
        y = g(x, 9, 3, 1) + e

        p0 = [0, 0, 0]
        np.random.seed(1)
        popt, fopt = pints.curve_fit(g, x, y, p0, method=pints.XNES)
        self.assertAlmostEqual(popt[0], 9, places=1)
        self.assertAlmostEqual(popt[1], 3, places=1)
        self.assertAlmostEqual(popt[2], 1, places=1)

        # Function must be callable
        self.assertRaisesRegex(
            ValueError, 'callable', pints.curve_fit, 3, x, y, p0)

        # Test with boundaries
        popt, fopt = pints.curve_fit(
            g, x, y, p0,
            boundaries=([-10, -10, -10], [10, 10, 10]), method=pints.XNES)
        self.assertAlmostEqual(popt[0], 9, places=1)
        self.assertAlmostEqual(popt[1], 3, places=1)
        self.assertAlmostEqual(popt[2], 1, places=1)

        # Test with parallelisation
        pints.curve_fit(g, x, y, p0, parallel=True, method=pints.XNES)

        # Test with invalid sizes of `x` and `y`
        x = np.linspace(-5, 5, 99)
        self.assertRaisesRegex(
            ValueError, 'dimension', pints.curve_fit, g, x, y, p0)


def f(x):
    """ Pickleable test function. """
    return (x[0] - 3) ** 2 + (x[1] + 5) ** 2


def g(x, a, b, c):
    """ Pickleable test function. """
    return a + b * x + c * x ** 2


if __name__ == '__main__':
    unittest.main()
