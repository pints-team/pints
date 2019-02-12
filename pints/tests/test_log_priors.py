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
import scipy.stats


class TestPrior(unittest.TestCase):

    def test_normal_prior(self):
        mean = 10
        std = 2
        p = pints.NormalLogPrior(mean, std)

        n = 10000
        r = 6 * np.sqrt(std)

        # Test left half of distribution
        x = np.linspace(mean - r, mean, n)
        px = [p([i]) for i in x]
        self.assertTrue(np.all(px[1:] >= px[:-1]))

        # Test right half of distribution
        y = np.linspace(mean, mean + std, n)
        py = [p([i]) for i in y]
        self.assertTrue(np.all(py[1:] <= py[:-1]))

        # Test derivatives
        x = [8]
        y, dy = p.evaluateS1(x)
        self.assertEqual(y, p(x))
        self.assertEqual(dy.shape, (1, ))
        self.assertEqual(dy[0], (mean - x[0]) / std**2)

        p = pints.NormalLogPrior(-1, 4.5)
        x = [3.75]
        self.assertAlmostEqual(p(x), -2.9801146954130457)
        p = pints.NormalLogPrior(10.4, 0.5)
        x = [5.5]
        y, dy = p.evaluateS1(x)
        self.assertAlmostEqual(y, -48.245791352644737)
        self.assertEqual(dy, 19.6)

    def test_normal_prior_sampling(self):
        mean = 10
        std = 2
        p = pints.NormalLogPrior(mean, std)

        d = 1
        n = 1
        x = p.sample(n)
        self.assertEqual(x.shape, (n, d))
        n = 10
        x = p.sample(n)
        self.assertEqual(x.shape, (n, d))

        # Very roughly check distribution (main checks are in numpy!)
        np.random.seed(1)
        p = pints.NormalLogPrior(mean, std)
        x = p.sample(10000)
        self.assertTrue(np.abs(mean - x.mean(axis=0)) < 0.1)
        self.assertTrue(np.abs(std - x.std(axis=0)) < 0.01)

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

        # Test errors
        self.assertRaises(ValueError, pints.ComposedLogPrior)
        self.assertRaises(ValueError, pints.ComposedLogPrior, 1)

        # Test derivatives
        p = pints.ComposedLogPrior(p1, p2)
        x = [8, -40]
        y, dy = p.evaluateS1(x)
        self.assertEqual(y, p(x))
        self.assertEqual(dy.shape, (2, ))
        y1, dy1 = p1.evaluateS1(x[:1])
        y2, dy2 = p2.evaluateS1(x[1:])
        self.assertAlmostEqual(dy[0], dy1[0])
        self.assertAlmostEqual(dy[1], dy2[0])

    def test_composed_prior_sampling(self):

        m1 = 10
        c1 = 2
        p1 = pints.NormalLogPrior(m1, c1)
        m2 = -50
        c2 = 100
        p2 = pints.NormalLogPrior(m2, c2)
        p = pints.ComposedLogPrior(p1, p2)

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
        d = p.n_parameters()
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

        # Test normal construction
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

        # Test from rectangular boundaries object
        b = pints.RectangularBoundaries(lower, upper)
        p = pints.UniformLogPrior(b)
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

        # Test custom boundaries object
        class CircleBoundaries(pints.Boundaries):
            def __init__(self, x, y, r):
                self.x, self.y, self.r = x, y, r

            def n_parameters(self):
                return 2

            def check(self, p):
                x, y = p
                return (x - self.x)**2 + (y - self.y)**2 < self.r**2

        b = CircleBoundaries(5, 5, 2)
        p = pints.UniformLogPrior(b)
        minf = -float('inf')
        self.assertTrue(p([0, 0]) == minf)
        self.assertTrue(p([4, 4]) > minf)

        # Test derivatives (always 0)
        for x in [[0, 0], [0, 5], [0, 19], [0, 21], [5, 0], [5, 21]]:
            y, dy = p.evaluateS1(x)
            self.assertEqual(y, p(x))
            self.assertEqual(dy.shape, (2, ))
            self.assertTrue(np.all(dy == 0))

        for x in [[1, 2], [1, 5], [1, 20 - 1e-14], [5, 5], [5, 20 - 1e-14]]:
            y, dy = p.evaluateS1(x)
            self.assertEqual(y, p(x))
            self.assertEqual(dy.shape, (2, ))
            self.assertTrue(np.all(dy == 0))

        # Test bad constructor
        self.assertRaises(ValueError, pints.UniformLogPrior, lower)

    def test_uniform_prior_sampling(self):
        lower = np.array([1, 2])
        upper = np.array([10, 20])
        p = pints.UniformLogPrior(lower, upper)

        # Test output formats
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

        # Roughly check distribution (main checks are in numpy!)
        np.random.seed(1)
        p = pints.UniformLogPrior(lower, upper)
        x = p.sample(10000)
        self.assertTrue(np.all(lower <= x))
        self.assertTrue(np.all(upper > x))
        self.assertTrue(
            np.linalg.norm(x.mean(axis=0) - 0.5 * (upper + lower)) < 0.1)

    def test_multivariate_normal_prior(self):
        # 1d test
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

        # 5d tests
        mean = [1, 2, 3, 4, 5]
        covariance = np.diag(mean)
        p = pints.MultivariateNormalLogPrior(mean, covariance)
        self.assertRaises(ValueError, p, [1, 2, 3])
        p([1, 2, 3, 4, 5])
        p([-1, 2, -3, 4, -5])

        # Test errors
        self.assertRaises(
            ValueError, pints.MultivariateNormalLogPrior, [1, 2],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_multivariate_normal_sampling(self):
        d = 1
        mean = 2
        covariance = [[1]]
        p = pints.MultivariateNormalLogPrior(mean, covariance)

        n = 1
        x = p.sample(n)
        self.assertEqual(x.shape, (n, d))
        n = 10
        x = p.sample(n)
        self.assertEqual(x.shape, (n, d))

        # 5d tests
        d = 5
        mean = np.array([1, 2, 3, 4, 5])
        covariance = np.diag(mean)
        p = pints.MultivariateNormalLogPrior(mean, covariance)
        n = 1
        x = p.sample(n)
        self.assertEqual(x.shape, (n, d))
        n = 10
        x = p.sample(n)
        self.assertEqual(x.shape, (n, d))

        # Roughly check distribution (main checks are in numpy!)
        np.random.seed(1)
        p = pints.MultivariateNormalLogPrior(mean, covariance)
        x = p.sample(10000)
        self.assertTrue(np.all(np.abs(mean - x.mean(axis=0)) < 0.1))
        self.assertTrue(np.all(
            np.abs(np.diag(covariance) - x.std(axis=0)**2) < 0.1))

    def test_student_t_prior(self):
        # Test two specific function values
        p1 = pints.StudentTLogPrior(0, 2, 10)
        self.assertEqual(p1([0]), -3.342305863833964)
        p2 = pints.StudentTLogPrior(10, 5, 10)
        self.assertEqual(p2([10]), -3.27120468204877)

        # Test exceptions
        self.assertRaises(ValueError, pints.StudentTLogPrior, 0, 0, 10)
        self.assertRaises(ValueError, pints.StudentTLogPrior, 0, -1, 10)
        self.assertRaises(ValueError, pints.StudentTLogPrior, 0, 1, 0)
        self.assertRaises(ValueError, pints.StudentTLogPrior, 0, 1, -1)

        # Test other function calls
        self.assertEqual(p1.n_parameters(), 1)
        self.assertEqual(p2.n_parameters(), 1)

    def test_student_t_prior_sampling(self):
        p1 = pints.StudentTLogPrior(0, 1000, 1)
        self.assertEqual(len(p1.sample()), 1)

        n = 10000
        samples1 = p1.sample(n)
        self.assertEqual(len(samples1), n)
        self.assertTrue(np.absolute(np.mean(samples1)) < 2)

        p2 = pints.StudentTLogPrior(0, 1, 1)
        samples2 = p2.sample(n)
        self.assertGreater(np.var(samples2), np.var(samples1))

        p3 = pints.StudentTLogPrior(0, 1000, 1000)
        samples3 = p3.sample(n)
        self.assertGreater(np.var(samples3), np.var(samples1))

        p4 = pints.StudentTLogPrior(1000, 1000, 1)
        samples4 = p4.sample(n)
        self.assertGreater(np.mean(samples4), np.mean(samples1))

    def test_cauchy_prior(self):
        # Test two specific function values
        p1 = pints.CauchyLogPrior(0, 10)
        self.assertEqual(p1([0]), -3.447314978843446)
        p2 = pints.CauchyLogPrior(10, 5)
        self.assertTrue(np.abs(p2([10]) + 2.7541677982835) < 0.001)

        # Test exceptions
        self.assertRaises(ValueError, pints.CauchyLogPrior, 0, 0)
        self.assertRaises(ValueError, pints.CauchyLogPrior, 10, -1)

        # Test other function calls
        self.assertEqual(p1.n_parameters(), 1)
        self.assertEqual(p2.n_parameters(), 1)

    def test_cauchy_prior_sampling(self):
        # Aren't many tests for Cauchy distributions
        # because they have no mean or variance!
        p1 = pints.CauchyLogPrior(0, 1000)
        self.assertEqual(len(p1.sample()), 1)
        self.assertEqual(len(p1.sample(100)), 100)

    def test_half_cauchy_prior(self):
        # Test two specific function values
        p1 = pints.HalfCauchyLogPrior(0, 10)
        self.assertEqual(p1([0]), -float('Inf'))
        self.assertAlmostEqual(p1([10]), -3.447314978843445)
        p2 = pints.HalfCauchyLogPrior(10, 5)
        self.assertAlmostEqual(p2([10]), -2.594487638427916)

        # Test exceptions
        self.assertRaises(ValueError, pints.HalfCauchyLogPrior, 0, 0)
        self.assertRaises(ValueError, pints.HalfCauchyLogPrior, 10, -1)

        # Test other function calls
        self.assertEqual(p1.n_parameters(), 1)
        self.assertEqual(p2.n_parameters(), 1)

    def test_half_cauchy_prior_sampling(self):
        # Aren't many tests for Cauchy distributions
        # because they have no mean or variance!
        p1 = pints.HalfCauchyLogPrior(0, 1000)
        self.assertEqual(len(p1.sample()), 1)
        n = 1000
        v_samples = p1.sample(n)
        self.assertEqual(len(v_samples), n)
        self.assertTrue(np.all(v_samples > 0))

    def test_beta_prior(self):

        # Test input parameters
        self.assertRaises(ValueError, pints.BetaLogPrior, 0, 0)
        self.assertRaises(ValueError, pints.BetaLogPrior, 2, -2)
        self.assertRaises(ValueError, pints.BetaLogPrior, -2, 2)

        p1 = pints.BetaLogPrior(0.123, 2.34)
        p2 = pints.BetaLogPrior(3.45, 4.56)

        points = [-2., 0.001, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999, 2.]

        # Test n_parameters
        self.assertEqual(p1.n_parameters(), 1)

        # Test specific points
        for point in points:
            to_test = [point]
            self.assertAlmostEqual(
                scipy.stats.beta.logpdf(to_test[0], 0.123, 2.34),
                p1(to_test),
                places=9)
            self.assertAlmostEqual(
                scipy.stats.beta.logpdf(to_test[0], 3.45, 4.56),
                p2(to_test),
                places=9)

        # Test derivatives
        p1_derivs = [0., -878.341341341342, -10.25888888888889,
                     -4.837619047619048,
                     -4.434, -5.719523809523809, -14.37444444444445,
                     -1340.877877877876,
                     0.]

        p2_derivs = [0., 2446.436436436437, 20.54444444444445,
                     3.080952380952382,
                     -2.219999999999999, -8.36666666666666, -32.87777777777778,
                     -3557.547547547544, 0.]

        for point, deriv in zip(points, p1_derivs):
            calc_val, calc_deriv = p1.evaluateS1([point])
            self.assertAlmostEqual(calc_deriv[0], deriv)

        for point, deriv in zip(points, p2_derivs):
            calc_val, calc_deriv = p2.evaluateS1([point])
            self.assertAlmostEqual(calc_deriv[0], deriv)

    def test_beta_prior_sampling(self):
        # Just returns samples from the numpy beta distribution so no utility
        # in verifying shape params - just check it's working as expected
        p1 = pints.BetaLogPrior(0.123, 2.34)
        self.assertEqual(len(p1.sample()), 1)

        n = 100
        samples1 = p1.sample(n)
        self.assertEqual(len(samples1), n)

    def test_exponential_prior(self):

        # Test input parameter
        self.assertRaises(ValueError, pints.ExponentialLogPrior, 0.0)
        self.assertRaises(ValueError, pints.ExponentialLogPrior, -1.0)

        r1 = 0.123
        r2 = 4.567

        p1 = pints.ExponentialLogPrior(r1)
        p2 = pints.ExponentialLogPrior(r2)

        points = [-2., 0.001, 0.1, 1.0, 2.45, 6.789]

        # Test n_parameters
        self.assertEqual(p1.n_parameters(), 1)

        # Test specific points
        for point in points:
            to_test = [point]
            self.assertAlmostEqual(
                scipy.stats.expon.logpdf(to_test[0], scale=1. / r1),
                p1(to_test), places=9)
            self.assertAlmostEqual(
                scipy.stats.expon.logpdf(to_test[0], scale=1. / r2),
                p2(to_test), places=9)

        # Test derivatives
        p1_derivs = [0., -r1, -r1, -r1, -r1]

        p2_derivs = [0., -r2, -r2, -r2, -r2]

        for point, deriv in zip(points, p1_derivs):
            calc_val, calc_deriv = p1.evaluateS1([point])
            self.assertAlmostEqual(calc_deriv[0], deriv)

        for point, deriv in zip(points, p2_derivs):
            calc_val, calc_deriv = p2.evaluateS1([point])
            self.assertAlmostEqual(calc_deriv[0], deriv)

    def test_exponential_prior_sampling(self):
        # Just returns samples from the numpy exponential distribution, but
        # because we are parameterising it with rate not shape, we check the
        # first moment to be sure we're doing the right thing
        p1 = pints.ExponentialLogPrior(0.25)
        self.assertEqual(len(p1.sample()), 1)

        n = 1000
        samples1 = p1.sample(n)
        self.assertEqual(len(samples1), n)

        # Mean should be ~ 1/0.25 = 4, so we check that this is very roughly
        # the case, but we can be very relaxed as we only check it's not ~0.25
        mean = np.mean(samples1).item()
        self.assertTrue(3. < mean < 4.)

    def test_gamma_prior(self):

        # Test input parameters
        self.assertRaises(ValueError, pints.GammaLogPrior, 0, 0)
        self.assertRaises(ValueError, pints.GammaLogPrior, 2, -2)
        self.assertRaises(ValueError, pints.GammaLogPrior, -2, 2)

        a1 = 0.123
        a2 = 4.567

        b1 = 2.345
        b2 = 0.356

        p1 = pints.GammaLogPrior(a1, b1)
        p2 = pints.GammaLogPrior(a2, b2)

        points = [-2., 0.001, 0.1, 1.0, 2.45, 6.789]

        # Test n_parameters
        self.assertEqual(p1.n_parameters(), 1)

        # Test specific points
        for point in points:
            to_test = [point]
            self.assertAlmostEqual(
                scipy.stats.gamma.logpdf(to_test[0], a=a1, scale=1. / b1),
                p1(to_test), places=9)
            self.assertAlmostEqual(
                scipy.stats.gamma.logpdf(to_test[0], a=a2, scale=1. / b2),
                p2(to_test), places=9)

        # Test derivatives
        p1_derivs = [0., -879.345, -11.115, -3.222, -2.70295918367347,
                     -2.474179555162763]

        p2_derivs = [0., 3566.643999999999, 35.314, 3.211, 1.099918367346939,
                     0.1694087494476359]

        for point, deriv in zip(points, p1_derivs):
            calc_val, calc_deriv = p1.evaluateS1([point])
            self.assertAlmostEqual(calc_deriv[0], deriv)

        for point, deriv in zip(points, p2_derivs):
            calc_val, calc_deriv = p2.evaluateS1([point])
            self.assertAlmostEqual(calc_deriv[0], deriv)

    def test_gamma_prior_sampling(self):
        # Just returns samples from the numpy gamma distribution, but because
        # we are parameterising it with rate not shape, we check the first
        # moment to be sure we're doing the right thing
        p1 = pints.GammaLogPrior(5.0, 0.25)
        self.assertEqual(len(p1.sample()), 1)

        n = 1000
        samples1 = p1.sample(n)
        self.assertEqual(len(samples1), n)

        # Mean should be ~ 5/0.25 = 20, so we check that this is very roughly
        # the case, but we can be very relaxed as we only check it's not ~1.25
        mean = np.mean(samples1).item()
        self.assertTrue(19. < mean < 20.)


if __name__ == '__main__':
    unittest.main()
