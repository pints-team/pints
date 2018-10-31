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

import pints.noise as pn


class TestNoise(unittest.TestCase):
    """
    Tests if the noise generators work ok.
    """
    def test_independent_noise(self):

        # Test on numpy vector, tuple shape
        clean = np.asarray([1, 2, 3, 10])
        noisy = clean + pn.independent(1, clean.shape)
        self.assertFalse(np.all(clean == noisy))

        # Test integer shape
        noise = pn.independent(3, 1000)

        # No need to test noise characteristics extensively: handled by numpy!
        np.random.seed(1)
        noise = pn.independent(3, 1000)
        self.assertTrue(np.abs(np.mean(noise)) < 0.2)
        self.assertTrue(np.abs(np.std(noise) - 3) < 0.3)

        # Test multidimensional arrays, single sigma
        noise = pn.independent(3, [10, 10])
        self.assertEqual(noise.shape, (10, 10))

        # Standard deviation cannot be 0 or less (handled by numpy)
        self.assertRaises(ValueError, pn.independent, -1, clean.shape)

        # Shape must be a nice shape (handled by numpy)
        self.assertRaises(TypeError, pn.independent, 1, 'hello')

    def test_ar1(self):

        # Simple test
        clean = np.array([1, 2, 3, 10, 15, 8])
        noisy = clean + pn.ar1(0.5, 5.0, len(clean))
        self.assertFalse(np.all(clean == noisy))

        # Test length
        self.assertEqual(len(pn.ar1(0.1, 1, 100)), 100)

        # Magnitude of rho must be less than 1
        pn.ar1(1, 5, 10)
        pn.ar1(-1, 5, 10)
        self.assertRaises(ValueError, pn.ar1, 1.1, 5, 10)
        self.assertRaises(ValueError, pn.ar1, -1.1, 5, 10)

        # Sigma cannot be negative
        pn.ar1(0.5, 5, 10)
        self.assertRaises(ValueError, pn.ar1, 0.5, -5, 10)

        # N cannot be negative
        pn.ar1(0.5, 5, 0)
        self.assertRaises(ValueError, pn.ar1, 0.5, 5, -1)

        # Test noise properties
        self.assertTrue(np.abs(np.std(pn.ar1(0.99, 1, 1000)) -
                               np.std(pn.ar1(0.50, 1, 1000)) < 5))
        self.assertTrue(np.abs(np.std(pn.ar1(0.50, 1, 1000)) -
                               np.std(pn.ar1(0.50, 5, 1000)) < 2))
        self.assertTrue(np.abs(np.mean(pn.ar1(-0.5, 1, 10000))) < 5)

    def test_ar1_unity(self):

        # Simle test
        clean = np.asarray([1.3, 2, 3, 10, 15, 8])
        noisy = clean + pn.ar1_unity(0.5, 5.0, len(clean))
        self.assertFalse(np.all(clean == noisy))

        # Test length
        self.assertEqual(len(pn.ar1_unity(0.1, 1, 100)), 100)

        # Magnitude of rho must be less than 1
        pn.ar1(1, 5, 10)
        pn.ar1(-1, 5, 10)
        self.assertRaises(ValueError, pn.ar1_unity, 1.1, 5, 10)
        self.assertRaises(ValueError, pn.ar1_unity, -1.1, 5, 10)

        # Sigma cannot be negative
        pn.ar1_unity(0.5, 5, 10)
        self.assertRaises(ValueError, pn.ar1_unity, 0.5, -5, 10)

        # N cannot be negative
        pn.ar1(0.5, 5, 0)
        self.assertRaises(ValueError, pn.ar1_unity, 0.5, 5, -1)

        # Test noise properties
        self.assertTrue(np.abs(np.std(pn.ar1_unity(0.9, 1, 10000)) -
                               np.std(pn.ar1_unity(0.50, 1, 10000))) < 2)
        self.assertTrue(np.abs(np.mean(pn.ar1_unity(-0.5, 1, 10000)) - 1) < 2)


if __name__ == '__main__':
    unittest.main()
