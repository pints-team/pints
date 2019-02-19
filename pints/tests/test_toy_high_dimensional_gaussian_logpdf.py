#!/usr/bin/env python3
#
# Tests the high-dimensional Gaussian log-pdf toy problem.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import pints.toy
import unittest
import numpy as np


class TestHighDimensionalGaussianLogPDF(unittest.TestCase):
    """
    Tests the high-dimensional Gaussian log-pdf toy problem.
    """
    def test_high_dimensional_log_pdf(self):

        # Test basic usage
        f = pints.toy.HighDimensionalGaussianLogPDF(3)
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

        f = pints.toy.HighDimensionalGaussianLogPDF(100)
        self.assertEqual(f.n_parameters(), 100)
        f1 = f(np.zeros(100))
        f2 = f(np.ones(100) * 0.1)
        self.assertTrue(np.isscalar(f1))
        self.assertTrue(np.isscalar(f2))
        self.assertTrue(f1 > f2)
        # Note: thorough testing of stats is done by scipy!

        # Test errors
        self.assertRaises(
            ValueError, pints.toy.HighDimensionalGaussianLogPDF, 0)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
