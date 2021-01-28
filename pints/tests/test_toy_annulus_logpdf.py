#!/usr/bin/env python3
#
# Tests the annulus distribution.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.toy
import unittest
import numpy as np


class TestAnnulusLogPDF(unittest.TestCase):
    """
    Tests the annulus log-pdf toy problems.
    """
    def test_default(self):

        # Default settings
        f = pints.toy.AnnulusLogPDF()
        self.assertEqual(f.n_parameters(), 2)
        self.assertEqual(f.r0(), 10)
        self.assertEqual(f.sigma(), 1)
        f1 = f([1, 1])
        f2 = f([10, 0])
        self.assertTrue(np.isscalar(f1))
        self.assertTrue(np.isscalar(f2))
        self.assertAlmostEqual(f1, -37.776802909473716)
        self.assertAlmostEqual(f2, -0.91893853320467267)
        self.assertAlmostEqual(f.mean_normed(), 10.099999999999993)
        self.assertAlmostEqual(f.var_normed(), 0.99000000000020805)
        self.assertAlmostEqual(f.moment_normed(3), 1060.3)
        a_mean = f.mean()
        self.assertEqual(a_mean[0], 0)
        self.assertEqual(a_mean[1], 0)

    def test_sample_and_moments(self):
        # Tests sampling

        # Change dimensions and beta
        f = pints.toy.AnnulusLogPDF(10, 15, 0.5)
        self.assertEqual(f.n_parameters(), 10)
        self.assertAlmostEqual(f.mean_normed(), 15.148688458505298)
        self.assertAlmostEqual(f.var_normed(), 0.24756486472776373)
        self.assertAlmostEqual(f.moment_normed(5), 806385.71374340181)

        x = f.sample(10)
        self.assertEqual(len(x), 10)
        f = pints.toy.AnnulusLogPDF()
        self.assertRaises(ValueError, f.__call__, [1])
        self.assertRaises(ValueError, f.__call__, [1, 2, 3])
        self.assertTrue(np.max(f.sample(1000)) < 100)
        self.assertRaises(ValueError, f.sample, 0)
        f = pints.toy.AnnulusLogPDF()
        samples = f.sample(100000)
        self.assertTrue(np.abs(np.mean(samples)) < 0.1)

        # Test _reject_sample_r
        self.assertTrue(np.all(f._reject_sample(100) > 0))

    def test_bad_constructors(self):
        self.assertRaises(
            ValueError, pints.toy.AnnulusLogPDF, 0, 1, 1)
        self.assertRaises(
            ValueError, pints.toy.AnnulusLogPDF, 1, 0, 1)
        self.assertRaises(
            ValueError, pints.toy.AnnulusLogPDF, 3, 1, -1)

    def test_suggested_bounds(self):
        # Tests suggested_bounds() method

        f = pints.toy.AnnulusLogPDF()
        bounds = f.suggested_bounds()
        a_val = 55
        self.assertTrue(np.array_equal([[-a_val, -a_val], [a_val, a_val]],
                                       bounds))
        r0 = 25
        dimensions = 5
        sigma = 20
        f = pints.toy.AnnulusLogPDF(dimensions=dimensions,
                                    r0=r0,
                                    sigma=sigma)
        bounds = f.suggested_bounds()
        r0_magnitude = (r0 + sigma) * (5**(1.0 / (dimensions - 1.0)))
        self.assertEqual(bounds[0][0], -r0_magnitude)
        self.assertEqual(bounds[1][0], r0_magnitude)
        self.assertTrue(np.array_equal(np.array(bounds).shape, [2, 5]))

    def test_distance_function(self):
        # Tests distance function

        log_pdf = pints.toy.AnnulusLogPDF()
        samples = log_pdf.sample(100)
        d = list(map(lambda x: np.linalg.norm(x), samples))
        dist = (np.abs(np.var(d) - log_pdf.var_normed()) +
                np.abs(np.mean(d) - log_pdf.mean_normed()))
        self.assertEqual(log_pdf.distance(samples), dist)
        f = log_pdf
        self.assertTrue(f.distance(samples) > 0)
        x = np.ones((100, 4))
        self.assertRaises(ValueError, f.distance, x)
        x = np.ones((100, 6))
        self.assertRaises(ValueError, f.distance, x)
        x = np.ones((100, 5, 2))
        self.assertRaises(ValueError, f.distance, x)

        log_pdf = pints.toy.AnnulusLogPDF(5, 20, 3)
        samples = log_pdf.sample(100)
        d = list(map(lambda x: np.linalg.norm(x), samples))
        dist = (np.abs(np.var(d) - log_pdf.var_normed()) +
                np.abs(np.mean(d) - log_pdf.mean_normed()))
        self.assertEqual(log_pdf.distance(samples), dist)
        f = log_pdf
        self.assertTrue(f.distance(samples) > 0)
        x = np.ones((100, 4))
        self.assertRaises(ValueError, f.distance, x)
        x = np.ones((100, 6))
        self.assertRaises(ValueError, f.distance, x)
        x = np.ones((100, 5, 2))
        self.assertRaises(ValueError, f.distance, x)

    def test_sensitivities(self):
        # Tests sensitivities

        f = pints.toy.AnnulusLogPDF()
        l, dl = f.evaluateS1([0, -9])
        self.assertEqual(l, f([0, -9]))
        self.assertEqual(len(dl), 2)
        self.assertEqual(dl[0], 0)
        self.assertEqual(dl[1], -1)
        f = pints.toy.AnnulusLogPDF(4, 20, 3)
        l, dl = f.evaluateS1([2, -1, 1, 3])
        self.assertEqual(l, f([2, -1, 1, 3]))
        self.assertEqual(len(dl), 4)
        top = 20 - np.sqrt(15)
        bottom = np.sqrt(15)
        self.assertAlmostEqual(dl[0], 2 * top / (9 * bottom))
        self.assertAlmostEqual(dl[1], -top / (9 * bottom))
        self.assertAlmostEqual(dl[2], top / (9 * bottom))
        self.assertAlmostEqual(dl[3], top / (3 * bottom))


if __name__ == '__main__':
    unittest.main()
