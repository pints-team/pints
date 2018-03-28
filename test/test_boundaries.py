#!/usr/bin/env python3
#
# Tests the Boundaries object.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import division
import unittest
import pints
import numpy as np


class TestBoundaries(unittest.TestCase):

    def test_boundaries(self):

        # Create boundaries
        pints.Boundaries([1, 2], [3, 4])
        pints.Boundaries([1], [2])
        pints.Boundaries(np.array([1, 2, 3]), [4, 5, 6])
        pints.Boundaries(1, 2)

        # Create invalid boundaries
        self.assertRaises(ValueError, pints.Boundaries, [1, 2], [1])
        self.assertRaises(ValueError, pints.Boundaries, [], [])
        self.assertRaises(ValueError, pints.Boundaries, [2], [1])
        self.assertRaises(ValueError, pints.Boundaries, [1, 1], [10, 1])

        # Check methods
        lower = [1, -2]
        upper = [3, 4]
        b = pints.Boundaries(lower, upper)
        self.assertEqual(b.n_parameters(), len(lower))
        self.assertTrue(np.all(b.lower() == np.array(lower)))
        self.assertTrue(np.all(b.upper() == np.array(upper)))
        self.assertTrue(np.all(b.range() == np.array(upper) - np.array(lower)))

        # Check checking
        # Within bounds
        self.assertTrue(b.check([2, 3]))
        # On a lower bound
        self.assertTrue(b.check([1, 3]))
        # Below a lower bound
        self.assertFalse(b.check([1 - 1e16, 4]))
        # On an upper bound
        self.assertFalse(b.check([3, 0]))
        # Above an upper bound
        self.assertFalse(b.check([2, 14]))
        # Wrong in every way
        self.assertFalse(b.check([-20, 20]))
        self.assertFalse(b.check([20, -20]))
        # Negative number
        self.assertFalse(b.check([2, -3]))

    def test_sampling(self):

        lower = np.array([1, -1])
        upper = np.array([2, 1])
        d = 2
        b = pints.Boundaries(lower, upper)
        self.assertTrue(b.check(b.sample()))

        n = 1
        x = b.sample()
        self.assertEqual(x.shape, (n, d))
        n = 10
        x = b.sample(n)
        self.assertEqual(x.shape, (n, d))

        for p in b.sample(50):
            self.assertTrue(b.check(p))


if __name__ == '__main__':
    unittest.main()
