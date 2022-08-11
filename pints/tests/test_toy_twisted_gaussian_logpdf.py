#!/usr/bin/env python3
#
# Tests the twisted gaussian logpdf toy distribution.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.toy
import unittest
import numpy as np


class TestTwistedGaussianLogPDF(unittest.TestCase):
    """
    Tests the twisted gaussian logpdf toy distribution.
    """
    def test_twisted_gaussian_logpdf(self):
        # Test TwistedGaussianLogPDF basics.

        # Test basics
        f = pints.toy.TwistedGaussianLogPDF()
        self.assertEqual(f.n_parameters(), 10)
        self.assertTrue(np.isscalar(f(np.zeros(10))))

        # Test errors
        self.assertRaises(ValueError, pints.toy.TwistedGaussianLogPDF, 1)
        self.assertRaises(ValueError, pints.toy.TwistedGaussianLogPDF, b=-1)

    def test_sampling_and_kl_divergence(self):
        # Test TwistedGaussianLogPDF.kl_divergence() and .sample().

        # Ensure consistent output
        np.random.seed(1)

        # Create banana LogPDFs
        d = 6
        log_pdf1 = pints.toy.TwistedGaussianLogPDF(d, 0.01, 90)
        log_pdf2 = pints.toy.TwistedGaussianLogPDF(d, 0.02, 80)
        log_pdf3 = pints.toy.TwistedGaussianLogPDF(d, 0.04, 100)

        # Sample from each
        n = 10000
        samples1 = log_pdf1.sample(n)
        samples2 = log_pdf2.sample(n)
        samples3 = log_pdf3.sample(n)

        # Compare calculated divergences
        # This also tests the "untwist" method.
        s11 = log_pdf1.kl_divergence(samples1)
        s12 = log_pdf1.kl_divergence(samples2)
        s13 = log_pdf1.kl_divergence(samples3)
        self.assertLess(s11, s12)
        self.assertLess(s11, s13)
        self.assertAlmostEqual(s11, 0.0012248323505286152)

        s21 = log_pdf2.kl_divergence(samples1)
        s22 = log_pdf2.kl_divergence(samples2)
        s23 = log_pdf2.kl_divergence(samples3)
        self.assertLess(s22, s21)
        self.assertLess(s22, s23)

        s31 = log_pdf3.kl_divergence(samples1)
        s32 = log_pdf3.kl_divergence(samples2)
        s33 = log_pdf3.kl_divergence(samples3)
        self.assertLess(s33, s32)
        self.assertLess(s33, s31)
        self.assertEqual(log_pdf3.kl_divergence(samples1),
                         log_pdf3.distance(samples1))

        # Test sample() errors
        self.assertRaises(ValueError, log_pdf1.sample, -1)

        # Test kl_divergence() errors
        self.assertEqual(samples1.shape, (n, d))
        x = np.ones((n, d + 1))
        self.assertRaises(ValueError, log_pdf1.kl_divergence, x)
        x = np.ones((n, d, 2))
        self.assertRaises(ValueError, log_pdf1.kl_divergence, x)

        # Test suggested bounds
        f = pints.toy.TwistedGaussianLogPDF()
        bounds = f.suggested_bounds()
        bounds1 = [[-50, 50], [-100, 100]]
        bounds1 = np.transpose(bounds1).tolist()
        self.assertTrue(np.array_equal(bounds, bounds1))

    def test_values_sensitivity(self):
        # Tests values of log pdf and sensitivities

        log_pdf = pints.toy.TwistedGaussianLogPDF(dimension=2)
        self.assertEqual(log_pdf([-20, -30]), -4.1604621594033908)
        x = [-1, 2]
        l, dl = log_pdf.evaluateS1(x)
        self.assertEqual(l, log_pdf(x))
        self.assertAlmostEqual(l, -35.3455121594)
        self.assertEqual(dl[0], -1.5799)
        self.assertEqual(dl[1], 7.9)

        # higher dimensions
        log_pdf = pints.toy.TwistedGaussianLogPDF(dimension=4, b=0.3, V=200)
        x = [-1, 2, -3, 12]
        l, dl = log_pdf.evaluateS1(x)
        self.assertAlmostEqual(l, -1747.4699253160925)
        self.assertEqual(l, log_pdf(x))
        self.assertAlmostEqual(dl[0], -34.619949999999996)
        self.assertAlmostEqual(dl[1], 57.699999999999996)
        self.assertEqual(dl[2], 3)
        self.assertEqual(dl[3], -12)


if __name__ == '__main__':
    unittest.main()
