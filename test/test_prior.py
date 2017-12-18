#!/usr/bin/env python3
#
# Tests Prior functions in Pints
#
from __future__ import division
import unittest


class TestPrior(unittest.TestCase):

    def test_normal_prior(self):
        return      # TODO

        import pints
        import numpy as np

        mean = 10
        cov = 2
        p = pints.NormalPrior(mean, cov)

        n = 10000
        r = 6 * np.sqrt(cov)
        w = float(r) / n

        # Test left half of distribution
        x = np.linspace(mean - r, mean, n)
        px = [p([i]) for i in x]
        self.assertTrue(np.all(px[1:] >= px[:-1]))
        self.assertAlmostEqual(np.sum(px) * w, 0.5, places=3)

        # Test right half of distribution
        y = np.linspace(mean, mean + r, n)
        py = [p([i]) for i in y]
        self.assertTrue(np.all(py[1:] <= py[:-1]))
        self.assertAlmostEqual(np.sum(py) * w, 0.5, places=3)

    def test_composed_prior(self):
        import pints
        import numpy as np

        m1 = 10
        c1 = 2
        p1 = pints.NormalPrior(m1, c1)

        m2 = -50
        c2 = 100
        p2 = pints.NormalPrior(m2, c2)

        p = pints.ComposedPrior(p1, p2)

        # Test at center
        peak1 = p1([m1])
        peak2 = p2([m2])
        self.assertEqual(p([m1, m2]), peak1 * peak2)

        # Test at random points
        np.random.seed(1)
        for i in range(100):
            x = np.random.normal(m1, c1)
            y = np.random.normal(m2, c2)
            self.assertAlmostEqual(p([x, y]), p1([x]) * p2([y]))

        # Test effect of increasing covariance
        p = [pints.ComposedPrior(
            p1, pints.NormalPrior(m2, c)) for c in range(1, 10)]
        p = [f([m1, m2]) for f in p]
        self.assertTrue(np.all(p[:-1] > p[1:]))

    def test_uniform_prior(self):
        import pints
        import numpy as np

        lower = np.array([1, 2])
        upper = np.array([10, 20])

        p = pints.UniformPrior(lower, upper)
        self.assertEqual(p([0, 0]), 0)
        self.assertEqual(p([0, 5]), 0)
        self.assertEqual(p([0, 20]), 0)
        self.assertEqual(p([0, 21]), 0)
        self.assertEqual(p([5, 0]), 0)
        self.assertEqual(p([5, 21]), 0)
        self.assertEqual(p([15, 0]), 0)
        self.assertEqual(p([15, 5]), 0)
        self.assertEqual(p([15, 20]), 0)
        self.assertEqual(p([15, 21]), 0)

        w = 1 / np.product(upper - lower)
        self.assertEqual(p([1, 2]), w)
        self.assertEqual(p([1, 5]), w)
        self.assertEqual(p([1, 20]), w)
        self.assertEqual(p([5, 5]), w)
        self.assertEqual(p([5, 20]), w)

# TODO Test MultiVariateNormalPrior


if __name__ == '__main__':
    unittest.main()
