#!/usr/bin/env python3
#
# Tests the noise generators
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
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
        self.assertRaisesRegex(
            ValueError, 'scale', pn.independent, -1, clean.shape)

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
        pn.ar1(0.9, 5, 10)
        pn.ar1(-0.9, 5, 10)
        self.assertRaisesRegex(ValueError, 'rho', pn.ar1, 1.1, 5, 10)
        self.assertRaisesRegex(ValueError, 'rho', pn.ar1, -1.1, 5, 10)

        # Sigma cannot be negative
        pn.ar1(0.5, 5, 10)
        self.assertRaisesRegex(
            ValueError, 'Standard deviation', pn.ar1, 0.5, -5, 10)

        # N cannot be negative
        pn.ar1(0.5, 5, 1)
        self.assertRaisesRegex(
            ValueError, 'Number of values', pn.ar1, 0.5, 5, 0)

        # Test noise properties
        self.assertTrue(np.abs(np.std(pn.ar1(0.99, 1, 1000)) -
                               np.std(pn.ar1(0.50, 1, 1000)) < 5))
        self.assertTrue(np.abs(np.std(pn.ar1(0.50, 1, 1000)) -
                               np.std(pn.ar1(0.50, 5, 1000)) < 2))
        self.assertTrue(np.abs(np.mean(pn.ar1(-0.5, 1, 10000))) < 5)

    def test_ar1_unity(self):

        # Simple test
        clean = np.asarray([1.3, 2, 3, 10, 15, 8])
        noisy = clean + pn.ar1_unity(0.5, 5.0, len(clean))
        self.assertFalse(np.all(clean == noisy))

        # Test length
        self.assertEqual(len(pn.ar1_unity(0.1, 1, 100)), 100)

        # Magnitude of rho must be less than 1
        pn.ar1(0.9, 5, 10)
        pn.ar1(-0.5, 5, 10)
        self.assertRaisesRegex(ValueError, 'rho', pn.ar1_unity, 1.1, 5, 10)
        self.assertRaisesRegex(ValueError, 'rho', pn.ar1_unity, -1.1, 5, 10)

        # Sigma cannot be negative
        pn.ar1_unity(0.5, 5, 10)
        self.assertRaisesRegex(
            ValueError, 'Standard deviation', pn.ar1_unity, 0.5, -5, 10)

        # N cannot be negative
        pn.ar1(0.5, 5, 1)
        self.assertRaisesRegex(
            ValueError, 'Number of values', pn.ar1_unity, 0.5, 5, 0)

        # Test noise properties
        self.assertTrue(np.abs(np.std(pn.ar1_unity(0.9, 1, 10000)) -
                               np.std(pn.ar1_unity(0.50, 1, 10000))) < 2)
        self.assertTrue(np.abs(np.mean(pn.ar1_unity(-0.5, 1, 10000)) - 1) < 2)

    def test_arma11(self):

        # Test construction errors
        self.assertRaisesRegex(
            ValueError, 'rho', pn.arma11, 1.1, 0.5, 5, 100)
        self.assertRaisesRegex(
            ValueError, 'theta', pn.arma11, 0.5, 1.1, 5, 100)
        self.assertRaisesRegex(
            ValueError, 'Standard deviation', pn.arma11, 0.5, 0.5, -5, 100)
        self.assertRaisesRegex(
            ValueError, 'Number of values', pn.arma11, 0.5, 0.5, 5, -100)

        # test values
        samples = pn.arma11(0.5, 0.5, 5, 10000)
        self.assertTrue(np.mean(samples) < 1)
        self.assertTrue(np.abs(np.std(samples) - 5) < 1)

    def test_arma11_unity(self):

        # Test construction errors
        self.assertRaisesRegex(
            ValueError, 'rho', pn.arma11_unity, 1.1, 0.5, 5, 100)
        self.assertRaisesRegex(
            ValueError, 'theta', pn.arma11_unity, 0.5, 1.1, 5, 100)
        self.assertRaisesRegex(
            ValueError, 'Standard dev', pn.arma11_unity, 0.5, 0.5, -5, 100)
        self.assertRaisesRegex(
            ValueError, 'Number of values', pn.arma11_unity, 0.5, 0.5, 5, -100)

        # test values
        samples = pn.arma11_unity(0.5, 0.5, 5, 10000)
        self.assertTrue(np.abs(np.mean(samples) - 1) < 1)
        self.assertTrue(np.abs(np.std(samples) - 5) < 1)

    def test_multiplicative_gaussian(self):

        # Test construction errors
        self.assertRaisesRegex(
            ValueError,
            'Standard deviation',
            pn.multiplicative_gaussian,
            1.0,
            -1.0,
            [1, 2, 3]
        )

        self.assertRaisesRegex(
            ValueError,
            'Standard deviation',
            pn.multiplicative_gaussian,
            1.0,
            [2.0, -1.0],
            np.array([[1, 2, 3], [4, 5, 6]])
        )

        f_too_many_dims = np.zeros((2, 10, 5))
        self.assertRaisesRegex(
            ValueError,
            'f must have be of shape',
            pn.multiplicative_gaussian,
            1.0,
            1.0,
            f_too_many_dims
        )

        self.assertRaisesRegex(
            ValueError,
            'eta must be',
            pn.multiplicative_gaussian,
            np.array([[1, 2, 3], [4, 5, 6]]),
            1.0,
            [1, 2, 3]
        )

        self.assertRaisesRegex(
            ValueError,
            'eta must be',
            pn.multiplicative_gaussian,
            np.array([1, 2, 3]),
            1.0,
            [1, 2, 3]
        )

        self.assertRaisesRegex(
            ValueError,
            'sigma must be',
            pn.multiplicative_gaussian,
            1.0,
            np.array([[1, 2, 3], [4, 5, 6]]),
            [1, 2, 3]
        )

        self.assertRaisesRegex(
            ValueError,
            'sigma must be',
            pn.multiplicative_gaussian,
            1.0,
            np.array([1, 2, 3]),
            [1, 2, 3]
        )

        # Test values
        samples_small_f = pn.multiplicative_gaussian(2.0, 1.0, [1] * 10000)
        self.assertTrue(np.abs(np.mean(samples_small_f)) < 1)
        self.assertTrue(np.abs(np.std(samples_small_f) - 1) < 1)

        samples_large_f = pn.multiplicative_gaussian(2.0, 1.0, [2] * 10000)
        self.assertTrue(np.abs(np.mean(samples_large_f)) < 1)
        self.assertTrue(np.abs(np.std(samples_large_f) - 4) < 1)

        # Test multi-outputs
        f_2d = np.array([[1, 2, 3, 4], [11, 12, 13, 14]])
        samples_2d_eta = pn.multiplicative_gaussian([1.0, 3.0], 5.0, f_2d)
        self.assertTrue(samples_2d_eta.shape == f_2d.shape)

        samples_2d_sigma = pn.multiplicative_gaussian(1.0, [0.5, 0.75], f_2d)
        self.assertTrue(samples_2d_sigma.shape == f_2d.shape)

        samples_2d_both = pn.multiplicative_gaussian([1.0, 3.0],
                                                     [0.5, 0.75], f_2d)
        self.assertTrue(samples_2d_both.shape == f_2d.shape)


if __name__ == '__main__':
    unittest.main()
