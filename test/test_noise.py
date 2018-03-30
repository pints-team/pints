#!/usr/bin/env python3
#
# Tests the noise generators
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import unittest
import numpy as np
from pints.noise import (add_independent_noise,
                         AR1,
                         add_AR1_noise,
                         AR1_unity,
                         multiply_AR1_noise)


class TestNoise(unittest.TestCase):
    """
    Tests if the noise generators work ok
    """

    def test_independent_noise(self):
        # Test the special case where the initial size is zero
        values_1 = np.asarray([1, 2, 3, 10])
        values_2 = add_independent_noise(values_1, 1.0)
        for v1, v2 in zip(values_1, values_2):
            self.assertNotEqual(v1, v2)
        self.assertRaises(ValueError, add_independent_noise, values_1, -1.0)
        self.assertRaises(ValueError, add_independent_noise, values_1, 0.0)
        self.assertRaises(ValueError, add_independent_noise, [], 10.0)
        self.assertGreater(np.std(add_independent_noise(values_1, 100.0)),
                           np.std(add_independent_noise(values_1, 0.1)))

    def test_AR1(self):
        self.assertRaises(ValueError, AR1, 10, 1, 100)
        self.assertRaises(ValueError, AR1, 0.5, -1, 100)
        self.assertRaises(TypeError, AR1, 0.5, 1, 100.5)
        self.assertRaises(ValueError, AR1, 0.5, 1, 0)
        self.assertTrue(np.abs(np.std(AR1(0.99, 1, 1000)) -
                        np.std(AR1(0.50, 1, 1000)) < 5))
        self.assertTrue(np.abs(np.std(AR1(0.50, 1, 1000)) -
                        np.std(AR1(0.50, 5, 1000)) < 2))
        self.assertEqual(len(AR1(-0.5, 1, 100)), 100)
        self.assertTrue(np.abs(np.mean(AR1(-0.5, 1, 10000))) < 5)

    def test_add_AR1_noise(self):
        values_1 = np.asarray([1, 2, 3, 10, 15, 8])
        values_2 = add_AR1_noise(values_1, 0.5, 5.0)
        for v1, v2 in zip(values_1, values_2):
            self.assertNotEqual(v1, v2)
        self.assertRaises(ValueError, add_AR1_noise, values_1, -100, 1)
        self.assertRaises(ValueError, add_AR1_noise, values_1, 0.33, 0)
        self.assertRaises(ValueError, add_AR1_noise, [], -0.25, 1)

    def test_AR1_unity(self):
        self.assertRaises(ValueError, AR1_unity, 10, 1, 100)
        self.assertRaises(ValueError, AR1_unity, 0.5, -1, 100)
        self.assertRaises(ValueError, AR1_unity, 0.5, 1, 0)
        self.assertTrue(np.abs(np.std(AR1_unity(0.9, 1, 10000)) -
                        np.std(AR1_unity(0.50, 1, 10000))) < 2)
        self.assertTrue(np.abs(np.mean(AR1_unity(-0.5, 1, 10000)) - 1) < 2)
        self.assertEqual(len(AR1_unity(-0.5, 1, 100)), 100)

    def test_multiply_AR1_noise(self):
        values_1 = np.asarray([1.3, 2, 3, 10, 15, 8])
        values_2 = multiply_AR1_noise(values_1, 0.5, 5.0)
        for v1, v2 in zip(values_1, values_2):
            self.assertNotEqual(v1, v2)
        self.assertRaises(ValueError, multiply_AR1_noise, values_1, 10, 1)
        self.assertRaises(ValueError, multiply_AR1_noise, values_1, 0.5, -1)
        self.assertRaises(ValueError, multiply_AR1_noise, [], 0.5, 1)


if __name__ == '__main__':
    unittest.main()
