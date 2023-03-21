#!/usr/bin/env python3
#
# Tests the multimodal Gaussian distribution.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.toy
import unittest
import numpy as np


class TestMultimodalGaussianLogPDF(unittest.TestCase):
    """
    Tests the multimodal log-pdf toy problems.
    """
    def test_basic(self):
        # Tests instantiations and calls in 2D and 3D across varying numbers of
        # modes.

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
        self.assertEqual(f([1e6, 1e6]), -np.inf)
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
                [[1, 0.1, 0], [0.1, 1, 0], [0, 0, 1]],
                [[1, 0, 0], [0, 1, 0.1], [0, 0.1, 1]],
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

    def test_bad_constructors(self):
        # Tests bad Constructors.

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

        # invalid covariance matrices
        self.assertRaises(
            ValueError, pints.toy.MultimodalGaussianLogPDF,
            [[1, 0], [1, 0]], [[[1, 0], [0, 1]], [[1, 0.1], [0, 1]]])
        self.assertRaises(
            ValueError, pints.toy.MultimodalGaussianLogPDF,
            [[1, 0], [1, 0]], [[[1, 0], [0, 1]], [[1, 1.1], [1.1, 1]]])

        with self.assertRaisesRegex(ValueError, 'Covariance matrices must'):
            pints.toy.MultimodalGaussianLogPDF(
                modes=[[10, 1], [10, 1], [10, 1]],
                covariances=[
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1]
                ]
            )

    def test_sensitivities(self):
        # Tests sensitivities.

        log_pdf = pints.toy.MultimodalGaussianLogPDF()
        l, dl = log_pdf.evaluateS1([1, 1])
        self.assertEqual(len(dl), 2)
        self.assertAlmostEqual(l, -2.8378770664093453)
        self.assertEqual(dl[0], -1)
        self.assertEqual(dl[1], -1)

        # non-standard
        covariances = [[[5**2, 0.5 * 5 * 3, -0.1 * 5 * 2],
                        [0.5 * 5 * 3, 3**2, 0.4 * 3 * 2],
                        [-0.1 * 5 * 2, 0.4 * 3 * 2, 2**2]],
                       [[5**2, 0.3 * 5 * 6, 0.2 * 5 * 7],
                        [0.3 * 5 * 6, 6**2, -0.3 * 6 * 7],
                        [0.2 * 5 * 7, -0.3 * 6 * 7, 7**2]]]
        log_pdf = pints.toy.MultimodalGaussianLogPDF(
            modes=[[1, 2, 3], [3, 4, 5]],
            covariances=covariances)
        l, dl = log_pdf.evaluateS1([2, 4, 5])
        self.assertEqual(len(dl), 3)
        self.assertAlmostEqual(l, -6.2170334785680836)
        self.assertAlmostEqual(dl[0], -0.024754331962513677)
        self.assertAlmostEqual(dl[1], -0.054933536830093638)
        self.assertAlmostEqual(dl[2], -0.39317158556789844)

    def test_suggested_bounds(self):
        # Tests suggested_bounds().

        f = pints.toy.MultimodalGaussianLogPDF()
        covariances = [[[5**2, 0.5 * 5 * 3, -0.1 * 5 * 2],
                        [0.5 * 5 * 3, 3**2, 0.4 * 3 * 2],
                        [-0.1 * 5 * 2, 0.4 * 3 * 2, 2**2]],
                       [[5**2, 0.3 * 5 * 6, 0.2 * 5 * 7],
                        [0.3 * 5 * 6, 6**2, -0.3 * 6 * 7],
                        [0.2 * 5 * 7, -0.3 * 6 * 7, 7**2]]]
        bounds = f.suggested_bounds()
        lower = -10
        upper = 20
        bounds1 = np.tile([lower, upper], (2, 1))
        bounds1 = np.transpose(bounds1).tolist()
        self.assertTrue(np.array_equal(bounds, bounds1))

        f = pints.toy.MultimodalGaussianLogPDF(
            modes=[[1, 2, 3], [3, 4, 5]],
            covariances=covariances)
        bounds = f.suggested_bounds()
        lower = 1 - 4
        upper = 5 + 4
        bounds1 = np.tile([lower, upper], (3, 1))
        bounds1 = np.transpose(bounds1).tolist()
        self.assertTrue(np.array_equal(bounds, bounds1))

    def test_distance_function(self):
        # Tests distance function which calls kl_divergence().

        f = pints.toy.MultimodalGaussianLogPDF()
        covariances = [[[5**2, 0.5 * 5 * 3, -0.1 * 5 * 2],
                        [0.5 * 5 * 3, 3**2, 0.4 * 3 * 2],
                        [-0.1 * 5 * 2, 0.4 * 3 * 2, 2**2]],
                       [[5**2, 0.3 * 5 * 6, 0.2 * 5 * 7],
                        [0.3 * 5 * 6, 6**2, -0.3 * 6 * 7],
                        [0.2 * 5 * 7, -0.3 * 6 * 7, 7**2]]]
        n = 1000
        d = f.n_parameters()
        samples1 = f.sample(n)
        self.assertEqual(samples1.shape, (n, d))
        kl = f.kl_divergence(samples1)
        for i in range(2):
            self.assertTrue(kl[i] > 0)
        self.assertEqual(np.sum(kl), f.distance(samples1))
        x = np.ones((n, d + 1))
        self.assertRaises(ValueError, f.kl_divergence, x)
        x = np.ones((n, d, 2))
        self.assertRaises(ValueError, f.kl_divergence, x)

        f = pints.toy.MultimodalGaussianLogPDF(
            modes=[[1, 2, 3], [3, 4, 5]],
            covariances=covariances)
        n = 1000
        d = f.n_parameters()
        samples1 = f.sample(n)
        self.assertEqual(samples1.shape, (n, d))
        kl = f.kl_divergence(samples1)
        for i in range(2):
            self.assertTrue(kl[i] > 0)
        self.assertEqual(np.sum(kl), f.distance(samples1))
        x = np.ones((n, d + 1))
        self.assertRaises(ValueError, f.kl_divergence, x)
        x = np.ones((n, d, 2))
        self.assertRaises(ValueError, f.kl_divergence, x)

        f = pints.toy.MultimodalGaussianLogPDF(modes=[[1], [2], [3]])
        n = 1000
        d = f.n_parameters()
        samples1 = f.sample(n)
        self.assertEqual(samples1.shape, (n, d))
        kl = f.kl_divergence(samples1)
        self.assertTrue(kl[0] > 0)
        self.assertEqual(np.sum(kl), f.distance(samples1))
        x = np.ones((n, d + 1))
        self.assertRaises(ValueError, f.kl_divergence, x)
        x = np.ones((n, d, 2))
        self.assertRaises(ValueError, f.kl_divergence, x)
        self.assertRaises(ValueError, f.sample, 0)

        # test KL divergence for when a mode is missed
        f = pints.toy.MultimodalGaussianLogPDF()
        samples = f.sample(n)
        covs = [[[0.1, 0], [0, 0.1]],
                [[0.1, 0], [0, 0.1]]]
        g = pints.toy.MultimodalGaussianLogPDF([[0, 0], [0, 0]], covs)
        samples1 = g.sample(n)
        self.assertTrue(f.distance(samples1) > f.distance(samples))


if __name__ == '__main__':
    unittest.main()
