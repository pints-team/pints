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
#import numpy as np


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
        pass


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
