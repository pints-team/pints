#!/usr/bin/env python3
#
# Tests the log likelihood classes.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import unittest
import pints
import pints.toy as toy
import numpy as np


class TestLogLikelihood(unittest.TestCase):
    def test_scaled_log_likelihood(self):

        model = toy.LogisticModel()
        real_parameters = [0.015, 500]
        test_parameters = [0.014, 501]
        sigma = 0.001
        times = np.linspace(0, 1000, 100)
        values = model.simulate(real_parameters, times)

        # Create an object with links to the model and time series
        problem = pints.SingleSeriesProblem(model, times, values)

        # Create a scaled and not scaled log_likelihood
        log_likelihood_not_scaled = pints.KnownNoiseLogLikelihood(
            problem, sigma)
        log_likelihood_scaled = pints.ScaledLogLikelihood(
            log_likelihood_not_scaled)

        eval_not_scaled = log_likelihood_not_scaled(test_parameters)
        eval_scaled = log_likelihood_scaled(test_parameters)

        self.assertEqual(int(eval_not_scaled), -20959169232)
        self.assertAlmostEqual(eval_scaled * len(times), eval_not_scaled)

    def test_known_and_unknown_noise_log_likelihood(self):

        model = toy.LogisticModel()
        parameters = [0.015, 500]
        sigma = 0.1
        times = np.linspace(0, 1000, 100)
        values = model.simulate(parameters, times)
        problem = pints.SingleSeriesProblem(model, times, values)

        # Test if known/unknown give same result
        l1 = pints.KnownNoiseLogLikelihood(problem, sigma)
        l2 = pints.UnknownNoiseLogLikelihood(problem)
        self.assertAlmostEqual(l1(parameters), l2(parameters + [sigma]))

    def test_sum_of_independent_log_likelihoods(self):
        model = toy.LogisticModel()
        x = [0.015, 500]
        sigma = 0.1
        times = np.linspace(0, 1000, 100)
        values = model.simulate(x, times)
        problem = pints.SingleSeriesProblem(model, times, values)

        l1 = pints.KnownNoiseLogLikelihood(problem, sigma)
        l2 = pints.UnknownNoiseLogLikelihood(problem)
        ll = pints.SumOfIndependentLogLikelihoods(l1, l1, l1)
        self.assertEqual(3 * l1(x), ll(x))

        # Test invalid constructors
        # Wrong number of arguments
        self.assertRaises(ValueError, pints.SumOfIndependentLogLikelihoods)
        self.assertRaises(ValueError, pints.SumOfIndependentLogLikelihoods, l1)
        # Wrong types
        self.assertRaises(
            ValueError, pints.SumOfIndependentLogLikelihoods, l1, 1)
        self.assertRaises(
            ValueError, pints.SumOfIndependentLogLikelihoods, problem, l1)
        # Mismatching dimensions
        self.assertRaises(
            ValueError, pints.SumOfIndependentLogLikelihoods, l1, l2)


if __name__ == '__main__':
    unittest.main()
