#!/usr/bin/env python3
#
# Tests the simple egg box toy LogPDF.
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
        """
        Tests :meth:`SimpleEggBoxLogPDF.kl_score()`.
        """
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
        s11 = log_pdf1.kl_score(samples1)
        s12 = log_pdf1.kl_score(samples2)
        self.assertLess(s11, s12)
        s21 = log_pdf2.kl_score(samples1)
        s22 = log_pdf2.kl_score(samples2)
        self.assertLess(s22, s21)

        # Test penalising if a mode is missing
        samples3 = np.vstack((
            samples2[samples2[:, 0] > 0],   # Top half
            samples2[samples2[:, 1] < 0],   # Left half
        ))
        s23 = log_pdf2.kl_score(samples3)
        self.assertLess(s22, s23)
        self.assertGreater(s23 / s22, 100)

        # Test sample arguments
        self.assertRaises(ValueError, log_pdf1.sample, -1)

        # Test shape testing
        self.assertEqual(samples1.shape, (n, 2))
        x = np.ones((n, 3))
        self.assertRaises(ValueError, log_pdf1.kl_score, x)
        x = np.ones((n, 2, 2))
        self.assertRaises(ValueError, log_pdf1.kl_score, x)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()

