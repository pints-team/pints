#!/usr/bin/env python3
#
# Tests Prior functions in Pints
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import division
import unittest
import pints
import numpy as np


class TestPrior(unittest.TestCase):

    def test_normal_prior(self):
        mean = 10
        cov = 2
        p = pints.NormalLogPrior(mean, cov)

        n = 10000
        r = 6 * np.sqrt(cov)

        # Test left half of distribution
        x = np.linspace(mean - r, mean, n)
        px = [p([i]) for i in x]
        self.assertTrue(np.all(px[1:] >= px[:-1]))

        # Test right half of distribution
        y = np.linspace(mean, mean + r, n)
        py = [p([i]) for i in y]
        self.assertTrue(np.all(py[1:] <= py[:-1]))

        # Test sampling
        d = 1
        n = 1
        x = p.sample(n)
        self.assertEqual(x.shape, (n, d))
        n = 10
        x = p.sample(n)
        self.assertEqual(x.shape, (n, d))

    def test_composed_prior(self):
        import pints
        import numpy as np

        m1 = 10
        c1 = 2
        p1 = pints.NormalLogPrior(m1, c1)

        m2 = -50
        c2 = 100
        p2 = pints.NormalLogPrior(m2, c2)

        p = pints.ComposedLogPrior(p1, p2)

        # Test at center
        peak1 = p1([m1])
        peak2 = p2([m2])
        self.assertEqual(p([m1, m2]), peak1 + peak2)

        # Test at random points
        np.random.seed(1)
        for i in range(100):
            x = np.random.normal(m1, c1)
            y = np.random.normal(m2, c2)
            self.assertAlmostEqual(p([x, y]), p1([x]) + p2([y]))

        # Test effect of increasing covariance
        p = [pints.ComposedLogPrior(
            p1, pints.NormalLogPrior(m2, c)) for c in range(1, 10)]
        p = [f([m1, m2]) for f in p]
        self.assertTrue(np.all(p[:-1] > p[1:]))

        # Test sampling
        p = pints.ComposedLogPrior(p1, p2)
        d = 2
        n = 1
        x = p.sample(n)
        self.assertEqual(x.shape, (n, d))
        n = 10
        x = p.sample(n)
        self.assertEqual(x.shape, (n, d))
        p = pints.ComposedLogPrior(
            p1,
            pints.MultivariateNormalLogPrior([0, 1, 2], np.diag([2, 4, 6])),
            p2,
            p2,
        )
        d = p.dimension()
        self.assertEqual(d, 6)
        n = 1
        x = p.sample(n)
        self.assertEqual(x.shape, (n, d))
        n = 10
        x = p.sample(n)
        self.assertEqual(x.shape, (n, d))

    def test_uniform_prior(self):
        lower = np.array([1, 2])
        upper = np.array([10, 20])

        p = pints.UniformLogPrior(lower, upper)
        m = float('-inf')
        self.assertEqual(p([0, 0]), m)
        self.assertEqual(p([0, 5]), m)
        self.assertEqual(p([0, 19]), m)
        self.assertEqual(p([0, 21]), m)
        self.assertEqual(p([5, 0]), m)
        self.assertEqual(p([5, 21]), m)
        self.assertEqual(p([15, 0]), m)
        self.assertEqual(p([15, 5]), m)
        self.assertEqual(p([15, 19]), m)
        self.assertEqual(p([15, 21]), m)
        self.assertEqual(p([10, 10]), m)
        self.assertEqual(p([5, 20]), m)

        w = -np.log(np.product(upper - lower))
        self.assertEqual(p([1, 2]), w)
        self.assertEqual(p([1, 5]), w)
        self.assertEqual(p([1, 20 - 1e-14]), w)
        self.assertEqual(p([5, 5]), w)
        self.assertEqual(p([5, 20 - 1e-14]), w)

        # Test sampling
        d = 2
        n = 1
        x = p.sample(n)
        self.assertEqual(x.shape, (n, d))
        n = 10
        x = p.sample(n)
        self.assertEqual(x.shape, (n, d))

        p = pints.UniformLogPrior([0], [1])
        d = 1
        n = 1
        x = p.sample(n)
        self.assertEqual(x.shape, (n, d))
        n = 10
        x = p.sample(n)
        self.assertEqual(x.shape, (n, d))

    def test_multivariate_normal_prior(self):
        # 1d test
        d = 1
        mean = 0
        covariance = 1

        # Input must be a matrix
        self.assertRaises(
            ValueError, pints.MultivariateNormalLogPrior, mean, covariance)
        covariance = [1]
        self.assertRaises(
            ValueError, pints.MultivariateNormalLogPrior, mean, covariance)

        # Basic test
        covariance = [[1]]
        p = pints.MultivariateNormalLogPrior(mean, covariance)
        p([0])
        p([-1])
        p([11])

        # Test sampling
        n = 1
        x = p.sample(n)
        self.assertEqual(x.shape, (n, d))
        n = 10
        x = p.sample(n)
        self.assertEqual(x.shape, (n, d))

        # 5d tests
        d = 5
        mean = [1, 2, 3, 4, 5]
        covariance = np.diag(mean)
        p = pints.MultivariateNormalLogPrior(mean, covariance)
        self.assertRaises(ValueError, p, [1, 2, 3])
        p([1, 2, 3, 4, 5])
        p([-1, 2, -3, 4, -5])

        # Test sampling in 5-d
        n = 1
        x = p.sample(n)
        self.assertEqual(x.shape, (n, d))
        n = 10
        x = p.sample(n)
        self.assertEqual(x.shape, (n, d))


if __name__ == '__main__':
    unittest.main()
