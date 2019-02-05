#!/usr/bin/env python3
#
# Tests the high-dimensional normal log-pdf toy problem.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import pints.toy
import unittest
import numpy as np
import scipy.stats


class TestHighDimensionalNormalLogPDF(unittest.TestCase):
    """
    Tests the high-dimensional normal log-pdf toy problem.
    """
    def test_high_dimensional_log_pdf(self):

        # Test basic usage
        f = pints.toy.HighDimensionalNormalLogPDF(3)
        self.assertEqual(f.n_parameters(), 3)
        cov = np.array([
            [1, 0.5 * np.sqrt(2), 0.5 * np.sqrt(3)],
            [0.5 * np.sqrt(2), 2, 0.5 * np.sqrt(2) * np.sqrt(3)],
            [0.5 * np.sqrt(3), 0.5 * np.sqrt(2) * np.sqrt(3), 3]])
        self.assertTrue(np.all(f._cov == cov))
        f1 = f([0, 0, 0])
        f2 = f([0.1, 0.1, 0.1])
        self.assertTrue(np.isscalar(f1))
        self.assertTrue(np.isscalar(f2))
        self.assertTrue(f1 > f2)

        f = pints.toy.HighDimensionalNormalLogPDF(100)
        self.assertEqual(f.n_parameters(), 100)
        f1 = f(np.zeros(100))
        f2 = f(np.ones(100) * 0.1)
        self.assertTrue(np.isscalar(f1))
        self.assertTrue(np.isscalar(f2))
        self.assertTrue(f1 > f2)

        # default
        f = pints.toy.HighDimensionalNormalLogPDF()
        self.assertEqual(f.n_parameters(), 20)
        self.assertEqual(f.rho(), 0.5)

        # change rho
        f = pints.toy.HighDimensionalNormalLogPDF(rho=-0.9)
        self.assertEqual(f.n_parameters(), 20)
        self.assertEqual(f.rho(), -0.9)

        # change both
        f = pints.toy.HighDimensionalNormalLogPDF(dimension=15,
                                                  rho=0.9)
        self.assertEqual(f.n_parameters(), 15)
        self.assertEqual(f.rho(), 0.9)

        # For 2d case check value versus Scipy (in case we change to
        # implementing via something other than Scipy)
        f = pints.toy.HighDimensionalNormalLogPDF(dimension=2)
        cov = [[1.0, np.sqrt(1.0 / 2.0)],
               [np.sqrt(1.0 / 2.0), 2.0]]
        mean = np.zeros(2)
        self.assertEqual(f([1, 2]), scipy.stats.multivariate_normal.logpdf(
            [1, 2], mean, cov))

        # check suggested bounds
        f = pints.toy.HighDimensionalNormalLogPDF(dimension=2)
        bounds = f.suggested_bounds()
        magnitude = 3 * np.sqrt(2.0)
        bounds1 = np.tile([-magnitude, magnitude], (2, 1))
        bounds1 = np.transpose(bounds).tolist()
        self.assertTrue(np.array_equal(bounds, bounds1))

        f = pints.toy.HighDimensionalNormalLogPDF()
        bounds = f.suggested_bounds()
        self.assertTrue(bounds[0][0], np.sqrt(20) * 3.0)

        # Test errors
        self.assertRaises(
            ValueError, pints.toy.HighDimensionalNormalLogPDF, 0)
        self.assertRaises(
            ValueError, pints.toy.HighDimensionalNormalLogPDF, 2, 2)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
