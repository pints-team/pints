#!/usr/bin/env python3
#
# Tests the basic methods of the CMA-ES optimiser.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import pints
import numpy as np


class TestVector(unittest.TestCase):
    """
    Tests conversion to a read-only vector type
    """
    def test_vector(self):
        # Test correct use with 1d arrays
        x = np.array([1, 2, 3])
        v = pints.vector(x)
        x = np.array([1])
        v = pints.vector(x)
        x = np.array([])
        v = pints.vector(x)

        # Test correct use with higher dimensional arrays
        x = np.array([1, 2, 3])
        w = pints.vector(x)
        x = x.reshape((3, 1, 1, 1, 1))
        v = pints.vector(x)
        self.assertTrue(np.all(w == v))
        x = x.reshape((1, 3, 1, 1, 1))
        v = pints.vector(x)
        self.assertTrue(np.all(w == v))
        x = x.reshape((1, 1, 1, 1, 3))
        v = pints.vector(x)
        self.assertTrue(np.all(w == v))

        # Test correct use with lists
        x = [1, 2, 3]
        v = pints.vector(x)
        self.assertTrue(np.all(w == v))
        x = [4]
        v = pints.vector(x)
        x = []
        v = pints.vector(x)

        # Test incorrect use with higher dimensional arrays
        x = np.array([4, 5, 6, 3, 2, 4])
        x = x.reshape((2, 3))
        self.assertRaises(ValueError, pints.vector, x)
        x = x.reshape((3, 2))
        self.assertRaises(ValueError, pints.vector, x)
        x = x.reshape((6, 1))
        v = pints.vector(x)

        # Test correct use with scalar
        x = 5
        v = pints.vector(x)
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0], 5)

        # Test read-only
        def assign():
            v[0] = 10
        self.assertRaises(ValueError, assign)


if __name__ == '__main__':
    unittest.main()
