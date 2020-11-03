#!/usr/bin/env python3
#
# Tests the Rosenbrock toy problems.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.toy
import unittest
import numpy as np


class TestRosenbrock(unittest.TestCase):
    """
    Tests the Rosenbrock toy problems.
    """
    def test_error(self):
        f = pints.toy.RosenbrockError()
        self.assertEqual(f.n_parameters(), 2)
        fx = f([10, 10])
        self.assertTrue(np.isscalar(fx))
        self.assertEqual(fx, 810081)

        xopt = f.optimum()
        fopt = f(xopt)
        self.assertEqual(fopt, 0)

        np.random.seed(1)
        for x in np.random.uniform(-5, 5, size=(10, 2)):
            self.assertTrue(f(x) > fopt)

    def test_log_pdf(self):
        f = pints.toy.RosenbrockLogPDF()
        self.assertEqual(f.n_parameters(), 2)
        fx = f([0.5, 6.0])
        self.assertTrue(np.isscalar(fx))
        self.assertAlmostEqual(fx, np.log(1.0 / 3307.5))

        xopt = f.optimum()
        fopt = f(xopt)
        self.assertEqual(fopt, 0)

        # sensitivity test
        l, dl = f.evaluateS1([3, 4])

        self.assertEqual(l, -np.log(2505))
        self.assertEqual(len(dl), 2)
        self.assertEqual(dl[0], float(-6004.0 / 2505.0))
        self.assertEqual(dl[1], float(200.0 / 501.0))

        # suggested bounds and distance measure
        bounds = f.suggested_bounds()
        bounds = [[-2, 4], [-1, 12]]
        bounds = np.transpose(bounds).tolist()
        self.assertTrue(np.array_equal(bounds, f.suggested_bounds()))

        x = np.ones((100, 3))
        self.assertRaises(ValueError, f.distance, x)
        x = np.ones((100, 3, 2))
        self.assertRaises(ValueError, f.distance, x)

        # there is no simple way to generate samples from Rosenbrock
        nsamples = 10000
        g = pints.toy.GaussianLogPDF([1, 1], [1, 1])
        samples = g.sample(nsamples)
        self.assertTrue(f.distance(samples) > 0)
        x = np.ones((100, 3))
        self.assertRaises(ValueError, f.distance, x)
        x = np.ones((100, 2, 2))
        self.assertRaises(ValueError, f.distance, x)

        # generate samples with mean and variance closer to true values
        g1 = pints.toy.GaussianLogPDF([0.86935785, 2.59978086],
                                      [[1.80537968, 2.70257559],
                                       [2.70257559, 8.52658308]])
        samples1 = g1.sample(nsamples)
        self.assertTrue(f.distance(samples1) > 0)
        self.assertTrue(f.distance(samples) > f.distance(samples1))


if __name__ == '__main__':
    unittest.main()
