#!/usr/bin/env python3
#
# Tests nested rejection sampler.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np

import pints
import pints.toy


class TestNestedRejectionSampler(unittest.TestCase):
    """
    Unit (not functional!) tests for :class:`NestedRejectionSampler`.
    """

    @classmethod
    def setUpClass(cls):
        """ Prepare for the test. """
        # Create toy model
        model = pints.toy.LogisticModel()
        cls.real_parameters = [0.015, 500]
        times = np.linspace(0, 1000, 1000)
        values = model.simulate(cls.real_parameters, times)

        # Add noise
        np.random.seed(1)
        cls.noise = 10
        values += np.random.normal(0, cls.noise, values.shape)
        cls.real_parameters.append(cls.noise)

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(model, times, values)

        # Create a uniform prior over both the parameters and the new noise
        # variable
        cls.log_prior = pints.UniformLogPrior(
            [0.01, 400],
            [0.02, 600]
        )

        # Create a log-likelihood
        cls.log_likelihood = pints.GaussianKnownSigmaLogLikelihood(
            problem, cls.noise)

    def test_construction_errors(self):
        # Tests if invalid constructor calls are picked up.

        # First arg must be a log likelihood
        self.assertRaisesRegex(
            ValueError, 'must extend pints.LogPrior',
            pints.NestedRejectionSampler, self.log_likelihood)

    def test_hyper_params(self):
        # Tests the hyper parameter interface is working.
        sampler = pints.NestedRejectionSampler(self.log_prior)
        self.assertEqual(sampler.n_hyper_parameters(), 1)
        sampler.set_hyper_parameters([220])

    def test_getters_and_setters(self):
        # Tests various get() and set() methods.
        sampler = pints.NestedRejectionSampler(self.log_prior)

        # Active points
        x = sampler.n_active_points() + 1
        self.assertNotEqual(sampler.n_active_points(), x)
        sampler.set_n_active_points(x)
        self.assertEqual(sampler.n_active_points(), x)
        self.assertRaisesRegex(
            ValueError, 'greater than 5', sampler.set_n_active_points, 5)
        self.assertEqual(sampler.name(), 'Nested rejection sampler')
        self.assertTrue(not sampler.needs_initial_phase())

    def test_ask(self):
        # Tests ask.
        sampler = pints.NestedRejectionSampler(self.log_prior)
        pts = sampler.ask(1)
        self.assertTrue(np.isfinite(self.log_likelihood(pts)))

        # test multiple points being asked and tell'd
        sampler = pints.NestedRejectionSampler(self.log_prior)
        pts = sampler.ask(50)
        self.assertEqual(len(pts), 50)
        fx = [self.log_likelihood(pt) for pt in pts]
        proposed = sampler.tell(fx)
        self.assertTrue(len(proposed) > 1)


if __name__ == '__main__':
    unittest.main()
