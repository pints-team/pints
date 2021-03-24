#!/usr/bin/env python3
#
# Tests the high-dimensional Gaussian log-pdf toy problem.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.toy
import unittest
import numpy as np
import scipy.stats


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

        # default
        f = pints.toy.HighDimensionalGaussianLogPDF()
        self.assertEqual(f.n_parameters(), 20)
        self.assertEqual(f.rho(), 0.5)

        # change rho
        f = pints.toy.HighDimensionalGaussianLogPDF(rho=0.9)
        self.assertEqual(f.n_parameters(), 20)
        self.assertEqual(f.rho(), 0.9)

        # change both
        f = pints.toy.HighDimensionalGaussianLogPDF(dimension=15,
                                                    rho=0.8)
        self.assertEqual(f.n_parameters(), 15)
        self.assertEqual(f.rho(), 0.8)

        # For 2d case check value versus Scipy (in case we change to
        # implementing via something other than Scipy)
        f = pints.toy.HighDimensionalGaussianLogPDF(dimension=2)
        cov = [[1.0, np.sqrt(1.0 / 2.0)],
               [np.sqrt(1.0 / 2.0), 2.0]]
        mean = np.zeros(2)
        self.assertEqual(
            f([1, 2]),
            scipy.stats.multivariate_normal.logpdf([1, 2], mean, cov))

        # check suggested bounds
        f = pints.toy.HighDimensionalGaussianLogPDF(dimension=2)
        bounds = f.suggested_bounds()
        magnitude = 3 * np.sqrt(2.0)
        bounds1 = np.tile([-magnitude, magnitude], (2, 1))
        bounds1 = np.transpose(bounds1).tolist()
        self.assertTrue(np.array_equal(bounds, bounds1))

        f = pints.toy.HighDimensionalGaussianLogPDF()
        bounds = f.suggested_bounds()
        self.assertTrue(bounds[0][0], np.sqrt(20) * 3.0)

        # Test kl_divergence() errors
        n = 1000
        d = f.n_parameters()
        samples1 = f.sample(n)
        self.assertEqual(samples1.shape, (n, d))
        x = np.ones((n, d + 1))
        self.assertRaises(ValueError, f.kl_divergence, x)
        x = np.ones((n, d, 2))
        self.assertRaises(ValueError, f.kl_divergence, x)
        self.assertTrue(f.kl_divergence(samples1) > 0)
        self.assertEqual(f.kl_divergence(samples1), f.distance(samples1))
        self.assertRaises(ValueError, f.sample, 0)

        # Test errors
        self.assertRaises(
            ValueError, pints.toy.HighDimensionalGaussianLogPDF, 0)
        self.assertRaises(
            ValueError, pints.toy.HighDimensionalGaussianLogPDF, 1)
        self.assertRaises(
            ValueError, pints.toy.HighDimensionalGaussianLogPDF, 2, 2)
        # in order for matrix to be positive definite there are bounds
        # on the lower value of rho > - 1 / (dims - 1)
        self.assertRaises(
            ValueError, pints.toy.HighDimensionalGaussianLogPDF, 4, -0.34)
        self.assertRaises(
            ValueError, pints.toy.HighDimensionalGaussianLogPDF, 11, -0.11)

    def test_sensitivities(self):
        # tests that sensitivities are correct

        f = pints.toy.HighDimensionalGaussianLogPDF(2)
        x = [1, 1]
        L, dL = f.evaluateS1(x)
        self.assertEqual(L, f(x))
        exact = [-0.8619288125423018, -0.19526214587563495]
        for i in range(len(exact)):
            self.assertAlmostEqual(dL[i], exact[i])


if __name__ == '__main__':
    unittest.main()
