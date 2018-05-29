#!/usr/bin/env python
#
# Tests the easy optimisation methods fmin and curve_fit.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import unittest
import numpy as np


class TestEasyOptimisation(unittest.TestCase):
    """
    Tests the easy optimisation methods fmin and curve_fit.
    """
    def test_fmin(self):
        """
        Tests :meth:`pints.fmin()`.
        """
        # Note: This just wraps around `Optimisation`, so testing done here is
        # for wrapper code, not main functionality!

        # Basic test
        def f(x):
            return (x[0] - 3) ** 2 + (x[1] + 5) ** 2

        xopt, fopt = pints.fmin(f, [1, 1], method=pints.XNES)
        self.assertAlmostEqual(xopt[0], 3)
        self.assertAlmostEqual(xopt[1], -5)

        # Function must be callable
        self.assertRaisesRegexp(ValueError, 'callable', pints.fmin, 3, [1])

        # Test with boundaries
        xopt, fopt = pints.fmin(
            f, [1, 1], boundaries=([-10, -10], [10, 10]), method=pints.XNES)
        self.assertAlmostEqual(xopt[0], 3)
        self.assertAlmostEqual(xopt[1], -5)

        # Test with extra arguments
        def g(x, y, z):
            return (x[0] - 3) ** 2 + (x[1] + 5) ** 2 + y / z
        xopt, fopt = pints.fmin(g, [1, 1], args=[1, 2], method=pints.XNES)
        self.assertAlmostEqual(xopt[0], 3)
        self.assertAlmostEqual(xopt[1], -5)

    def test_curve_fit(self):
        """
        Tests :meth:`pints.curve_fit()`.
        """
        # Note: This just wraps around `Optimisation`, so testing done here is
        # for wrapper code, not main functionality!
        np.random.seed(1)

        # Basic test
        def f(x, a, b, c):
            return a + b * x + c * x ** 2

        x = np.linspace(-5, 5, 100)
        e = np.random.normal(loc=0, scale=0.1, size=x.shape)
        y = f(x, 9, 3, 1) + e

        p0 = [0, 0, 0]
        popt = pints.curve_fit(f, x, y, p0, method=pints.XNES)
        self.assertTrue(np.abs(popt[0] - 9) < 0.1)
        self.assertTrue(np.abs(popt[1] - 3) < 0.1)
        self.assertTrue(np.abs(popt[2] - 1) < 0.1)

        # Function must be callable
        self.assertRaisesRegexp(
            ValueError, 'callable', pints.curve_fit, 3, x, y, p0)

        # Test with boundaries
        pints.curve_fit(
            f, x, y, p0,
            boundaries=([-10, -10, -10], [10, 10, 10]), method=pints.XNES)
        self.assertTrue(np.abs(popt[0] - 9) < 0.1)
        self.assertTrue(np.abs(popt[1] - 3) < 0.1)
        self.assertTrue(np.abs(popt[2] - 1) < 0.1)

        # Test with invalid sizes of `x` and `y`
        x = np.linspace(-5, 5, 99)
        self.assertRaisesRegexp(
            ValueError, 'dimension', pints.curve_fit, f, x, y, p0)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
