#!/usr/bin/env python3
#
# Tests the twisted gaussian logpdf toy distribution.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
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
        """
        Test TwistedGaussianLogPDF basics.
        """
        # Test basics
        f = pints.toy.TwistedGaussianLogPDF()
        self.assertEqual(f.n_parameters(), 10)
        self.assertTrue(np.isscalar(f(np.zeros(10))))

        # Test errors
        self.assertRaises(
            ValueError, pints.toy.TwistedGaussianLogPDF, 1)
        self.assertRaises(
            ValueError, pints.toy.TwistedGaussianLogPDF, b=-1)

    def test_sampling_and_kl_divergence(self):
        """
        Test TwistedGaussianLogPDF.kl_divergence() and .sample().
        """
        # Ensure consistent output
        #np.random.seed(1)

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
        s11 = log_pdf1.kl_divergence(samples1)
        s12 = log_pdf1.kl_divergence(samples2)
        s13 = log_pdf1.kl_divergence(samples3)
        self.assertLess(s11, s12)
        self.assertLess(s11, s13)

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

        # Test sample() errors
        self.assertRaises(ValueError, log_pdf1.sample, -1)

        # Test kl_divergence() errors
        self.assertEqual(samples1.shape, (n, d))
        x = np.ones((n, d + 1))
        self.assertRaises(ValueError, log_pdf1.kl_divergence, x)
        x = np.ones((n, d, 2))
        self.assertRaises(ValueError, log_pdf1.kl_divergence, x)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
