#!/usr/bin/env python
#
# Tests Prior functions in Pints
#
import unittest
class TestPrior(unittest.TestCase):
    def test_normal_prior(self):
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
        
        peak1 = p1([m1])
        peak2 = p2([m2])
        self.assertEqual(p([m1, m2]), peak1 * peak2)
        #TODO Add more tests

#TODO Test MultiVariateNormalPrior
#TODO Test UniformPrior

if __name__ == '__main__':
    unittest.main()
