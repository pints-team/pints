#!/usr/bin/env python3
#
# Tests the multimodal normal distribution.
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


class TestMultimodalGaussianLogPDF(unittest.TestCase):
    """
    Tests the multimodal log-pdf toy problems.
    """
    def test_default(self):

        # Default settings
        f = pints.toy.MultimodalGaussianLogPDF()
        self.assertEqual(f.n_parameters(), 2)
        f1 = f([0, 0])
        f2 = f([10, 10])
        self.assertTrue(np.isscalar(f1))
        self.assertTrue(np.isscalar(f2))
        self.assertEqual(f1, f2)
        f3 = f([0.1, 0])
        self.assertTrue(f3 < f1)
        f4 = f([0, -0.1])
        self.assertTrue(f4 < f1)
        self.assertEqual(f([1e6, 1e6]), -float('inf'))
        # Note: This is very basic testing, real tests are done in scipy!

        # Single mode, 3d, standard covariance
        f = pints.toy.MultimodalGaussianLogPDF([[1, 1, 1]])
        self.assertEqual(f.n_parameters(), 3)
        f1 = f([1, 1, 1])
        f2 = f([0, 0, 0])
        self.assertTrue(np.isscalar(f1))
        self.assertTrue(np.isscalar(f2))
        self.assertTrue(f1 > f2)

        # Three modes, non-standard covariance
        f = pints.toy.MultimodalGaussianLogPDF(
            modes=[[1, 1, 1], [10, 10, 10], [20, 20, 20]],
            covariances=[
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[1, 1, 0], [.1, 1, 0], [0, 0, 1]],
                [[1, 0, 0], [0, 1, 1], [0, 0, 1]],
            ]
        )
        self.assertEqual(f.n_parameters(), 3)
        f1 = f([1, 1, 1])
        f2 = f([0, 0, 0])
        self.assertTrue(np.isscalar(f1))
        self.assertTrue(np.isscalar(f2))
        self.assertTrue(f1 > f2)

        # More modes than dimensions
        pints.toy.MultimodalGaussianLogPDF([
            [1, 1], [1.5, 1.5], [3, 0], [0, 3.5]
        ])

        # Bad constructors
        self.assertRaises(
            ValueError, pints.toy.MultimodalGaussianLogPDF, [])
        self.assertRaises(
            ValueError, pints.toy.MultimodalGaussianLogPDF, [[1], [1, 2]])
        pints.toy.MultimodalGaussianLogPDF(
            None, [[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
        self.assertRaises(
            ValueError, pints.toy.MultimodalGaussianLogPDF, None,
            [[[1, 0], [0, 1]]])
        self.assertRaises(
            ValueError, pints.toy.MultimodalGaussianLogPDF, None,
            [[[1, 0], [0]], [[1, 0], [0, 1]]])


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
