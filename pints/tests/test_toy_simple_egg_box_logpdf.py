#!/usr/bin/env python3
#
# Tests the simple egg box toy LogPDF.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.toy
import unittest
import numpy as np


class TestSimpleEggBoxLogPDF(unittest.TestCase):
    """
    Tests the simple egg box logpdf toy distribution.
    """
    def test_simple_egg_box_logpdf(self):
        # Test basics
        f = pints.toy.SimpleEggBoxLogPDF()
        self.assertEqual(f.n_parameters(), 2)
        self.assertTrue(np.isscalar(f(np.zeros(2))))

        # Test construction errors
        self.assertRaises(
            ValueError, pints.toy.SimpleEggBoxLogPDF, sigma=0)
        self.assertRaises(
            ValueError, pints.toy.SimpleEggBoxLogPDF, r=0)

    def test_sampling_and_divergence(self):
        # Tests :meth:`SimpleEggBoxLogPDF.kl_divergence()`.

        # Ensure consistent output
        np.random.seed(1)

        # Create some log pdfs
        log_pdf1 = pints.toy.SimpleEggBoxLogPDF(2, 4)
        log_pdf2 = pints.toy.SimpleEggBoxLogPDF(3, 6)

        # Generate samples from each
        n = 100
        samples1 = log_pdf1.sample(n)
        samples2 = log_pdf2.sample(n)

        # Test divergence scores
        s11 = log_pdf1.kl_divergence(samples1)
        s12 = log_pdf1.kl_divergence(samples2)
        self.assertLess(s11, s12)
        s21 = log_pdf2.kl_divergence(samples1)
        s22 = log_pdf2.kl_divergence(samples2)
        self.assertLess(s22, s21)

        # Test penalising if a mode is missing
        samples3 = np.vstack((
            samples2[samples2[:, 0] > 0],   # Top half
            samples2[samples2[:, 1] < 0],   # Left half
        ))
        s23 = log_pdf2.kl_divergence(samples3)
        self.assertLess(s22, s23)
        self.assertGreater(s23 / s22, 100)

        # Test sample arguments
        self.assertRaises(ValueError, log_pdf1.sample, -1)

        # Test shape testing
        self.assertEqual(samples1.shape, (n, 2))
        x = np.ones((n, 3))
        self.assertRaises(ValueError, log_pdf1.kl_divergence, x)
        x = np.ones((n, 2, 2))
        self.assertRaises(ValueError, log_pdf1.kl_divergence, x)

    def test_sensitivity_bounds_distance(self):
        # Tests :meth:`SimpleEggBoxLogPDF.evaluateS1()`,
        # :meth:`SimpleEggBoxLogPDF.suggested_bounds()` and
        # :meth:`SimpleEggBoxLogPDF.distance()`

        f = pints.toy.SimpleEggBoxLogPDF()
        l, dl = f.evaluateS1([-5, 2])
        self.assertEqual(l, f([-5, 2]))
        self.assertAlmostEqual(l, -13.781024134434123)
        self.assertAlmostEqual(dl[0], -1.5)
        self.assertAlmostEqual(dl[1], 2.9999991)
        self.assertTrue(np.array_equal(f.suggested_bounds(),
                                       [[-16.0, -16.0], [16.0, 16.0]]))
        samples = f.sample(100)
        self.assertTrue(f.kl_divergence(samples) > 0)
        self.assertEqual(f.kl_divergence(samples), f.distance(samples))

        f = pints.toy.SimpleEggBoxLogPDF(3, 5)
        l, dl = f.evaluateS1([-1, -7])
        self.assertEqual(l, f([-1, -7]))
        self.assertAlmostEqual(l, -46.269777289511559)
        self.assertAlmostEqual(dl[0], -4.6662126879796366)
        self.assertAlmostEqual(dl[1], -2.6666666666666639)
        self.assertTrue(np.array_equal(f.suggested_bounds(),
                                       [[-30.0, -30.0], [30.0, 30.0]]))
        samples = f.sample(100)
        self.assertTrue(f.kl_divergence(samples) > 0)
        self.assertEqual(f.kl_divergence(samples), f.distance(samples))


if __name__ == '__main__':
    unittest.main()
