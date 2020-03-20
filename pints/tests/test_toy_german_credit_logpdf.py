#!/usr/bin/env python3
#
# Tests the German credit toy distribution.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.toy
import unittest
import numpy as np


class TestGermanCreditLogPDF(unittest.TestCase):
    """
    Tests the logpdf toy distribution from fitting a logistic model to German
    credit data.
    """
    @classmethod
    def setUpClass(cls):
        """ Set up problem for tests. """
        # download data
        model = pints.toy.GermanCreditLogPDF(download=True)
        x, y = model.data()
        cls.y = y
        cls.x = x
        cls.model = model

    def test_download(self):
        # tests that method can download data from UCI repo
        x, y = self.model.data()
        self.assertEqual(x.shape[0], 1000)
        self.assertEqual(x.shape[1], 25)
        self.assertEqual(len(y), 1000)

    def test_errors(self):
        # tests errors of inapropriate function calls and inits
        self.assertRaises(ValueError, pints.toy.GermanCreditLogPDF,
                          np.zeros((27, 27)), self.y)
        self.assertRaises(ValueError, pints.toy.GermanCreditLogPDF,
                          self.x, np.ones(1000) * 2)
        self.assertRaises(ValueError, pints.toy.GermanCreditLogPDF,
                          self.x, self.y, True)
        self.assertRaises(ValueError, pints.toy.GermanCreditLogPDF,
                          None, self.y)
        self.assertRaises(ValueError, pints.toy.GermanCreditLogPDF,
                          self.x, None)

    def test_local(self):
        # tests that model can be instantiated using local files
        x, y = self.model.data()
        model = pints.toy.GermanCreditLogPDF(x=x, y=y)
        x1, y1 = model.data()
        self.assertTrue(np.array_equal(x, x1))
        self.assertTrue(np.array_equal(y, y1))

    def test_values(self):
        # tests calls
        self.assertAlmostEqual(self.model(np.zeros(25)), -693.1471805599322,
                               places=6)
        self.assertAlmostEqual(self.model(np.ones(25)), -2887.6292678483533,
                               places=6)

    def test_sensitivities(self):
        # test sensitivity values vs reference
        val, dp = self.model.evaluateS1(np.zeros(25))
        self.assertEqual(val, self.model(np.zeros(25)))
        self.assertEqual(len(dp), 25)
        self.assertEqual(dp[0], -200)
        self.assertAlmostEqual(dp[1], -160.7785147438439, places=6)

    def test_givens(self):
        # tests whether boundaries are correct and n_parameters
        self.assertEqual(25, self.model.n_parameters())
        borders = self.model.suggested_bounds()
        self.assertEqual(borders[0][0], -100)
        self.assertEqual(borders[1][0], 100)


if __name__ == '__main__':
    unittest.main()
