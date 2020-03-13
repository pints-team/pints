#!/usr/bin/env python
#
# Tests the German credit toy distribution.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import pints.toy
import unittest
import numpy as np
import io
import urllib
import urllib.request
from scipy import stats


class TestGermanCreditHierarchicalLogPDF(unittest.TestCase):
    """
    Tests the logpdf toy distribution from fitting a hierarchical logistic
    model to German credit data.
    """
    @classmethod
    def setUpClass(cls):
        """ Set up problem for tests. """
        # download data
        url="http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric" # noqa
        with urllib.request.urlopen(url) as url:
            raw_data = url.read()
        a = np.genfromtxt(io.BytesIO(raw_data), delimiter=4)[:, :25]

        # get output
        y = a[:, -1]
        y[y == 1] = -1
        y[y == 2] = 1
        cls.y = y

        # get inputs and standardise
        x = a[:, :-1]
        x = stats.zscore(x)
        x1 = np.zeros((x.shape[0], x.shape[1] + 1))
        x1[:, 0] = np.ones(x.shape[0])
        x1[:, 1:] = x
        x = np.copy(x1)
        cls.x = x

        cls.model = pints.toy.GermanCreditHierarchicalLogPDF(x, y)

    def test_download(self):
        # tests that method can download data from UCI repo
        model = pints.toy.GermanCreditHierarchicalLogPDF(download=True)
        x, y, z = model.data()
        self.assertTrue(np.array_equal(x, self.x))
        self.assertTrue(np.array_equal(y, self.y))

    def test_errors(self):
        # tests errors of inapropriate function calls and inits
        self.assertRaises(ValueError, pints.toy.GermanCreditHierarchicalLogPDF,
                          np.zeros((27, 27)), self.y)
        self.assertRaises(ValueError, pints.toy.GermanCreditHierarchicalLogPDF,
                          self.x, np.ones(1000) * 2)
        self.assertRaises(ValueError, pints.toy.GermanCreditHierarchicalLogPDF,
                          self.x, self.y, True)
        self.assertRaises(ValueError, pints.toy.GermanCreditHierarchicalLogPDF,
                          None, self.y)
        self.assertRaises(ValueError, pints.toy.GermanCreditHierarchicalLogPDF,
                          self.x, None)

    def test_values(self):
        # tests calls
        self.assertAlmostEqual(self.model(np.ones(326)),
                               -20174.077700157857,
                               places=6)

    def test_sensitivities(self):
        # test sensitivity values vs reference
        val, dp = self.model.evaluateS1(np.ones(326))
        self.assertEqual(val, self.model(np.ones(326)))
        self.assertEqual(len(dp), 326)
        self.assertAlmostEqual(dp[0], -1000.02)
        self.assertAlmostEqual(dp[1], -700.8386959844057, places=6)

    def test_givens(self):
        # tests whether boundaries are correct and n_parameters
        self.assertEqual(326, self.model.n_parameters())
        borders = self.model.suggested_bounds()
        self.assertEqual(borders[0][0], -100)
        self.assertEqual(borders[1][0], 100)


if __name__ == '__main__':
    unittest.main()
