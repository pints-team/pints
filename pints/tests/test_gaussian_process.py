#!/usr/bin/env python3
#
# Tests the basic methods of the CMAES optimiser.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import unittest
import numpy as np

import pints
import pints.toy
import matplotlib.pyplot as plt

debug = False

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestGaussianProcess(unittest.TestCase):
    """
    Tests the basic methods of the gaussian process log pdf.
    """
    def setUp(self):
        """ Called before every test """
        np.random.seed(1)

    def problem1D(self):
        """ Returns a test problem. """
        d = 1
        mean = np.array([3.0])
        sigma = np.array([[1]])
        log_pdf = pints.toy.GaussianLogPDF(mean, sigma)

        return log_pdf

    def problem2D(self):
        """ Returns a test problem. """
        d = 2
        mean = np.array([3.0, -3.0])
        sigma = np.array([[1, 0], [0, 1]])
        log_pdf = pints.toy.GaussianLogPDF(mean, sigma)

        return log_pdf


    def test_fitting(self):
        """ fits the gp to the problem. """
        log_pdf = self.problem1D()

        n = 100
        samples = log_pdf.sample(n)
        values = log_pdf(samples)  + np.random.normal(0,0.1,size=n)
        gp = pints.GaussianProcess(samples, values)
        test_samples = np.sort(log_pdf.sample(n).reshape(-1,1),axis=0)
        test_values = [gp.predict(test_samples[i,:]) for i in range(test_samples.shape[0])]
        test_means = [mv[0] for mv in test_values]
        test_stddev = np.sqrt([mv[1] for mv in test_values])

        plt.figure()
        plt.scatter(samples, values, label='original')
        plt.plot(test_samples, test_means, label='gp')
        plt.fill_between(test_samples.reshape(-1),test_means-test_stddev,test_means+test_stddev,alpha=0.3)
        plt.show()

if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
