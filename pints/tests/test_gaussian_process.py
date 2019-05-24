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

    def test_likelihood_gradient(self):
        log_pdf = self.problem1D()
        n = 100
        samples = log_pdf.sample(n)
        values = log_pdf(samples) + np.random.normal(0, 0.1, size=n)
        gp = pints.GaussianProcess(samples, values)
        p1_values = np.linspace(0.4, 0.6, 50)
        p2_values = np.linspace(0.4, 0.6, 50)
        p3_values = np.linspace(0.4, 0.6, 50)
        p1_likelihood = np.empty_like(p1_values)
        p2_likelihood = np.empty_like(p2_values)
        p3_likelihood = np.empty_like(p3_values)
        p1_grad_likelihood = np.empty_like(p1_values)
        p2_grad_likelihood = np.empty_like(p2_values)
        p3_grad_likelihood = np.empty_like(p3_values)
        centre_point = [0.5, 0.5, 0.5]
        for i, value in enumerate(p1_values):
            gp.set_hyper_parameters([value, centre_point[1], centre_point[2]])
            p1_likelihood[i] = gp.likelihood()
            p1_grad_likelihood[i] = gp.grad_likelihood()[0]

        for i, value in enumerate(p2_values):
            gp.set_hyper_parameters([centre_point[0], value, centre_point[2]])
            p2_likelihood[i] = gp.likelihood()
            p2_grad_likelihood[i] = gp.grad_likelihood()[1]

        for i, value in enumerate(p3_values):
            gp.set_hyper_parameters([centre_point[0], centre_point[1], value])
            p3_likelihood[i] = gp.likelihood()
            p3_grad_likelihood[i] = gp.grad_likelihood()[2]

        self.assertLess(
            np.linalg.norm((p1_grad_likelihood[1:-1] -
                            np.gradient(p1_likelihood, p1_values)[1:-1])) /
            np.linalg.norm(p1_grad_likelihood[1:-1]),
            5e-3
        )
        self.assertLess(
            np.linalg.norm((p2_grad_likelihood[1:-1] -
                            np.gradient(p2_likelihood, p2_values)[1:-1])) /
            np.linalg.norm(p2_grad_likelihood[1:-1]),
            5e-3
        )
        self.assertLess(
            np.linalg.norm((p3_grad_likelihood[1:-1] -
                            np.gradient(p3_likelihood, p3_values)[1:-1])) /
            np.linalg.norm(p3_grad_likelihood[1:-1]),
            5e-3
        )

    def test_approximate_likelihood(self):
        log_pdf = self.problem1D()
        n = 100
        samples = log_pdf.sample(n)
        values = log_pdf(samples) + np.random.normal(0, 0.1, size=n)
        gp_standard = pints.GaussianProcess(samples, values)
        gp_free = pints.GaussianProcess(samples, values, matrix_free=True)
        gp_dense = pints.GaussianProcess(samples, values, dense_matrix=True)
        gp_h2 = pints.GaussianProcess(samples, values, hierarchical_matrix=True)

        for gp in [gp_standard, gp_free, gp_dense, gp_h2]:
            gp.set_hyper_parameters([0.5, 0.5, 0.5])

        gp_h2._gaussian_process.set_h2_order(4)

        grad_likelihood_exact = gp_standard.grad_likelihood()
        likelihood_exact = gp_standard.likelihood()
        for gp in [gp_dense, gp_free, gp_h2]:
            gp._gaussian_process.set_stochastic_samples(4)
            gp._gaussian_process.set_tolerance(1e-6)
            gp._gaussian_process.set_chebyshev_n(100)
            repeats = 50
            likelihood_approx = np.empty(repeats, dtype=float)
            grad_likelihood_approx = np.empty((repeats, 3), dtype=float)
            for r in range(repeats):
                likelihood_approx[r] = gp.likelihood()
                grad_likelihood_approx[r, :] = gp.grad_likelihood()

            self.assertAlmostEqual(likelihood_exact, np.mean(likelihood_approx),
                                   delta=np.std(likelihood_approx))

            for i in range(3):
                self.assertAlmostEqual(
                    grad_likelihood_exact[i], np.mean(grad_likelihood_approx[:, i]),
                    delta=np.std(grad_likelihood_approx[:, i]))

    def test_direct_likelihood(self):
        log_pdf = self.problem1D()
        n = 100
        samples = log_pdf.sample(n)
        values = log_pdf(samples) + np.random.normal(0, 0.1, size=n)
        gp_standard = pints.GaussianProcess(samples, values)
        gp_direct = pints.GaussianProcess(samples, values, direct=True)

        for gp in [gp_standard, gp_direct]:
            gp.set_hyper_parameters([0.5, 0.5, 0.5])

        grad_likelihood_exact = gp_standard.grad_likelihood()
        likelihood_exact = gp_standard.likelihood()
        for gp in [gp_direct]:
            likelihood_direct = gp.likelihood()
            grad_likelihood_direct = gp.grad_likelihood()

            np.testing.assert_almost_equal(
                likelihood_exact, likelihood_direct, decimal=7)

            np.testing.assert_almost_equal(
                grad_likelihood_exact, grad_likelihood_direct, decimal=7)

    def test_predict(self):
        log_pdf = self.problem1D()
        n = 100
        samples = log_pdf.sample(n)
        values = log_pdf(samples) + np.random.normal(0, 0.1, size=n)
        gp_standard = pints.GaussianProcess(samples, values)
        gp_free = pints.GaussianProcess(samples, values, matrix_free=True)
        gp_dense = pints.GaussianProcess(samples, values, dense_matrix=True)
        gp_direct = pints.GaussianProcess(samples, values, direct=True)

        for gp in [gp_standard, gp_free, gp_dense, gp_direct]:
            gp.set_hyper_parameters([0.5, 0.5, 0.5])

        x = np.array([0.5*(samples[0] + samples[1])])
        mean_exact, var_exact = gp_standard.predict(x)
        self.assertEqual(mean_exact, gp_standard(x))
        for gp in [gp_free, gp_dense, gp_direct]:
            mean_approx, var_approx = gp.predict(x)
            self.assertAlmostEqual(mean_approx, gp(x))
            self.assertAlmostEqual(mean_approx, mean_exact)
            self.assertAlmostEqual(var_approx, var_exact)

    def test_fitting(self):
        """ fits the gp to the problem. """
        for log_pdf in [self.problem1D(), self.problem2D()]:

            n = 20
            samples = log_pdf.sample(n)
            values = log_pdf(samples) + np.random.normal(0, 0.1, size=n)
            for gp in [
                    pints.GaussianProcess(samples, values),
                    pints.GaussianProcess(samples, values, direct=True),
                    pints.GaussianProcess(samples, values, matrix_free=True),
                    pints.GaussianProcess(samples, values, dense_matrix=True),
            ]:
                gp.optimise_hyper_parameters()

                test_samples = samples
                if len(test_samples.shape) == 1:
                    test_samples = test_samples.reshape(-1, 1)

                test_values = np.empty((test_samples.shape[0], 2))
                for i in range(test_samples.shape[0]):
                    test_values[i, :] = gp.predict(test_samples[i, :])
                test_means = test_values[:, 0]
                test_stddev = np.sqrt(test_values[:, 1])

                # check 95% confidence intervals
                failure = np.logical_or(
                    values > (test_means + 1.96 * test_stddev),
                    values < (test_means - 1.96 * test_stddev)
                )
                self.assertLess(np.sum(failure), 0.05*n)

                # check mean against log_pdf
                true_means = log_pdf(samples)
                self.assertLess(np.linalg.norm(test_means-true_means) /
                                np.linalg.norm(true_means), 5e-2)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
