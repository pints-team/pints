#!/usr/bin/env python3
#
# Tests the sampling initialisation method
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#

import pints
import pints.io
import pints.toy
import unittest
import numpy as np


class TestMCMCInitialisationMethod(unittest.TestCase):
    """
    Tests `sample_initial_points` method for generating random initial starting
    locations.
    """
    @classmethod
    def setUpClass(cls):
        """ Prepare problem for tests. """
        # Load a forward model
        model = pints.toy.LogisticModel()

        # Create some toy data
        real_parameters = [0.015, 500]
        times = np.linspace(0, 1000, 1000)
        org_values = model.simulate(real_parameters, times)

        # Add noise
        noise = 10
        values = org_values + np.random.normal(0, noise, org_values.shape)
        real_parameters = np.array(real_parameters + [noise])

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(model, times, values)

        # Create an error measure
        cls.score = pints.SumOfSquaresError(problem)
        cls.boundaries = pints.RectangularBoundaries(
            [0, 400], [0.05, 600])

        # Create a log-likelihood function (adds an extra parameter!)
        log_likelihood = pints.GaussianLogLikelihood(problem)

        # Create a uniform prior over both the parameters and the new noise
        cls.log_prior = pints.UniformLogPrior(
            [0.01, 400, noise * 0.1],
            [0.02, 600, noise * 100]
        )

        # Create a posterior log-likelihood (log(likelihood * prior))
        cls.log_posterior = pints.LogPosterior(log_likelihood, cls.log_prior)

    def test_default_initialisation(self):
        # tests that log_prior can be used for initial sampling

        nchains = 1
        xs = pints.sample_initial_points(self.log_posterior, nchains)
        self.assertEqual(len(xs), nchains)
        [self.assertTrue(np.isfinite(self.log_posterior(x))) for x in xs]

        nchains = 4
        xs = pints.sample_initial_points(self.log_posterior, nchains)
        self.assertEqual(len(xs), nchains)
        [self.assertTrue(np.isfinite(self.log_posterior(x))) for x in xs]

        # check parallel initialisation works
        xs = pints.sample_initial_points(self.log_posterior, nchains,
                                         parallel=True)
        self.assertEqual(len(xs), nchains)
        xs = pints.sample_initial_points(self.log_posterior, nchains,
                                         parallel=True, n_workers=2)
        self.assertEqual(len(xs), nchains)

    def test_errormeasure_initialisation(self):
        # tests that initialisation works ok for error measures

        nchains = 4
        xs = pints.sample_initial_points(self.score, nchains,
                                         boundaries=self.boundaries)
        self.assertEqual(len(xs), nchains)
        # check points within boundaries
        for x in xs:
            self.assertTrue(self.boundaries.check(x))

    def test_errors(self):
        # tests errors when calling method with wrong inputs

        # test passing a callable but not pints object
        def f_test(theta):
            return sum(theta)
        nchains = 4
        self.assertRaisesRegex(
            ValueError, 'either pints.LogPDF or pints.ErrorMeasure.',
            pints.sample_initial_points, f_test, nchains, self.log_prior)

        # pass a non-callable object as random_sampler
        nchains = 4
        self.assertRaisesRegex(
            ValueError, 'random_sampler must be a callable function',
            pints.sample_initial_points,
            self.log_posterior, nchains,
            [0.015, 500, 10] * nchains)

        # try non log-posterior without passing random_sampler
        log_pdf = pints.toy.GaussianLogPDF()
        self.assertRaisesRegex(
            ValueError, 'random_sampler must be supplied',
            pints.sample_initial_points, log_pdf, nchains)
        self.assertRaisesRegex(
            ValueError, 'random_sampler must be supplied',
            pints.sample_initial_points, self.score, nchains)

        # boundaries aren't of right type of object
        boundaries = log_pdf
        self.assertRaisesRegex(
            ValueError, 'must be a pints.Boundaries',
            pints.sample_initial_points,
            log_pdf, nchains, boundaries=boundaries)

        # boundaries aren't correct dimensions for posterior
        self.assertRaisesRegex(
            ValueError, 'boundaries must match dimension',
            pints.sample_initial_points,
            self.log_posterior, nchains, boundaries=self.boundaries)

        # n_chains < 1?
        self.assertRaisesRegex(
            ValueError, 'points must be 1 or more',
            pints.sample_initial_points, self.log_posterior, 0.5)

    def test_bespoke_initialisation(self):
        # test using user-specified initialisation function

        # test that different initialisation produces different starting dist
        nchains = 4
        noise = 10
        xs = pints.sample_initial_points(self.log_posterior, nchains)
        log_prior1 = pints.UniformLogPrior(
            [0.0199, 599.99, noise * 99.99],
            [0.02, 600, noise * 100]
        )
        xs1 = pints.sample_initial_points(self.log_posterior, nchains,
                                          log_prior1.sample)
        self.assertTrue(sum(np.vstack(xs).mean(axis=0) <=
                            np.vstack(xs1).mean(axis=0)) == 3)
        for x in xs:
            self.assertTrue(np.isfinite(self.log_posterior(x)))

        # test initialisation for log_pdf (non-log-posterior)
        log_pdf = pints.toy.GaussianLogPDF()
        log_pdf1 = pints.toy.GaussianLogPDF(mean=[1, 1], sigma=[2, 2])
        init_sampler = log_pdf1.sample
        xs = pints.sample_initial_points(log_pdf, nchains, init_sampler)
        for x in xs:
            self.assertTrue(np.isfinite(log_pdf(x)))

    def test_initialisation_fails(self):
        # tests that initialisation can fail in specified number of tries

        from scipy.stats import multivariate_normal
        noise = 10
        nchains = 4

        def init_sampler(n_chains):
            return multivariate_normal.rvs(mean=[0.015, 500, noise],
                                           cov=np.diag([10, 10000, noise]),
                                           size=nchains)

        self.assertRaisesRegex(
            RuntimeError, 'Initialisation failed since function not finite',
            pints.sample_initial_points,
            self.log_posterior, nchains, init_sampler, max_tries=2)


if __name__ == '__main__':
    unittest.main()
