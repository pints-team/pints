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
        # Test basics
        f = pints.toy.TwistedGaussianLogPDF()
        self.assertEqual(f.n_parameters(), 10)
        self.assertTrue(np.isscalar(f(np.zeros(10))))

        # TODO: Test more?

        # Test errors
        self.assertRaises(
            ValueError, pints.toy.TwistedGaussianLogPDF, 1)
        self.assertRaises(
            ValueError, pints.toy.TwistedGaussianLogPDF, b=-1)

    def test_kl_divergence(self):
        """
        Tests :meth:`TwistedGaussianLogPDF.kl_divergence()`.
        """
        # Ensure consistent output
        np.random.seed(1)

        # Create a banana distribution LogPDF
        dimension = 6
        b = 0.15
        V = 90
        log_pdf = pints.toy.TwistedGaussianLogPDF(dimension, b, V)

        # Generate samples from similar distributions
        def sample(n, dimension, b, V):
            x = np.random.randn(n, 2)
            x[:, 0] *= np.sqrt(V)
            x[:, 1] += b * (x[:, 0] ** 2 - V)
            if dimension > 2:
                x = np.hstack((x, np.random.randn(n, dimension - 2)))
            return x

        #dimension = 2
        #x = sample(10000, dimension, b, V)
        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.scatter(x[:, 0], x[:, 1])
        #plt.show()

        # Test divergence of various distributions
        n = 1000
        s1 = sample(n, dimension, b, V)
        s2 = s1 + 0.03
        s3 = s2 * 1.01
        dkl1 = log_pdf.kl_divergence(s1)
        dkl2 = log_pdf.kl_divergence(s2)
        dkl3 = log_pdf.kl_divergence(s3)
        self.assertLess(dkl1, dkl2)
        self.assertLess(dkl2, dkl3)

        # Test shape testing
        self.assertEqual(s1.shape, (n, dimension))
        x = np.ones((n, dimension + 1))
        self.assertRaises(ValueError, log_pdf.kl_divergence, x)
        x = np.ones((n, dimension, 2))
        self.assertRaises(ValueError, log_pdf.kl_divergence, x)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
