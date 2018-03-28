#!/usr/bin/env python3
#
# Tests the basic methods of the CMA-ES optimiser.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import unittest
import pints
import numpy as np


class TestMatrix2d(unittest.TestCase):
    """
    Tests conversion to a read-only 2-d matrix type
    """
    def test_matrix2d(self):
        # Test correct use with 2d arrays
        x = np.array([[1, 2], [2, 3], [4, 3]])
        v = pints.matrix2d(x)
        x = np.array([[1, 2, 3], [2, 3, 4]])
        v = pints.matrix2d(x)

        # Test correct use with 1d arrays
        x = np.array([1, 2, 3, 4]).reshape((4, 1))
        v = pints.matrix2d(x)
        self.assertEqual(v.shape, (4, 1))
        x = np.array([1, 2, 3, 4])
        v = pints.matrix2d(x)
        self.assertEqual(v.shape, (4, 1))

        # Test correct use with lists
        x = [[1, 2], [2, 3], [4, 3]]
        v = pints.matrix2d(x)
        x = [[1, 2, 3], [2, 3, 4]]
        v = pints.matrix2d(x)

        # Test incorrect use with higher dimensional arrays
        x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        self.assertTrue(x.ndim == 3)
        self.assertRaises(ValueError, pints.matrix2d, x)

        # Test read-only
        def assign():
            v[0, 0] = 10
        self.assertRaises(ValueError, assign)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
