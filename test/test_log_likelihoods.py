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

        # Test bad constructor
        self.assertRaises(ValueError, pints.ScaledLogLikelihood, model)

    def test_known_and_unknown_noise_log_likelihood(self):

        model = toy.LogisticModel()
        parameters = [0.015, 500]
        sigma = 0.1
        times = np.linspace(0, 1000, 100)
        values = model.simulate(parameters, times)
        values += np.random.normal(0, sigma, values.shape)
        problem = pints.SingleSeriesProblem(model, times, values)

        # Test if known/unknown give same result
        l1 = pints.KnownNoiseLogLikelihood(problem, sigma)
        l2 = pints.UnknownNoiseLogLikelihood(problem)
        self.assertAlmostEqual(l1(parameters), l2(parameters + [sigma]))

        # Test invalid constructors
        self.assertRaises(
            ValueError, pints.KnownNoiseLogLikelihood, problem, 0)
        self.assertRaises(
            ValueError, pints.KnownNoiseLogLikelihood, problem, -1)

    def test_known_and_unknown_multivariate_noise_log_likelihood(self):

        model = toy.FitzhughNagumoModel()
        parameters = [0.5, 0.5, 0.5]
        sigma = 0.1
        times = np.linspace(0, 100, 100)
        values = model.simulate(parameters, times)
        values += np.random.normal(0, sigma, values.shape)
        problem = pints.MultiSeriesProblem(model, times, values)

        # Test if known/unknown give same result
        l1 = pints.KnownMultivariateNoiseLogLikelihood(problem, sigma)
        l2 = pints.UnknownMultivariateNoiseLogLikelihood(problem)
        self.assertAlmostEqual(
            l1(parameters),
            l2(parameters + [sigma, sigma]))

        # Test invalid constructors
        self.assertRaises(
            ValueError, pints.KnownMultivariateNoiseLogLikelihood, problem, 0)
        self.assertRaises(
            ValueError, pints.KnownMultivariateNoiseLogLikelihood, problem, -1)

    def test_known_normal_and_multivariate_noise(self):

        # Define boring 1-output and 2-output models
        class NullModel1(pints.ForwardModel):
            def dimension(self):
                return 1

            def simulate(self, x, times):
                return np.zeros(times.shape)

        class NullModel2(pints.ForwardModel):
            def dimension(self):
                return 1

            def n_outputs(self):
                return 2

            def simulate(self, x, times):
                return np.zeros((len(times), 2))

        # Create two single output problems
        times = np.arange(10)
        np.random.seed(1)
        sigma1 = 3
        sigma2 = 5
        values1 = np.random.uniform(0, sigma1, times.shape)
        values2 = np.random.uniform(0, sigma2, times.shape)
        model1d = NullModel1()
        problem1 = pints.SingleSeriesProblem(model1d, times, values1)
        problem2 = pints.SingleSeriesProblem(model1d, times, values2)
        log1 = pints.KnownNoiseLogLikelihood(problem1, sigma1)
        log2 = pints.KnownNoiseLogLikelihood(problem2, sigma2)

        # Create one multi output problem
        values3 = np.array([values1, values2]).swapaxes(0, 1)
        model2d = NullModel2()
        problem3 = pints.MultiSeriesProblem(model2d, times, values3)
        log3 = pints.KnownMultivariateNoiseLogLikelihood(
            problem3, [sigma1, sigma2])

        # Check if we get the right output
        self.assertAlmostEqual(log1(0) + log2(0), log3(0))

    def test_sum_of_independent_log_likelihoods(self):
        model = toy.LogisticModel()
        x = [0.015, 500]
        sigma = 0.1
        times = np.linspace(0, 1000, 100)
        values = model.simulate(x, times)
        problem = pints.SingleSeriesProblem(model, times, values)

        l1 = pints.KnownNoiseLogLikelihood(problem, sigma)
        l2 = pints.UnknownNoiseLogLikelihood(problem)
        ll = pints.SumOfIndependentLogLikelihoods([l1, l1, l1])
        self.assertEqual(l1.dimension(), ll.dimension())
        self.assertEqual(3 * l1(x), ll(x))

        # Test invalid constructors
        # Wrong number of arguments
        self.assertRaises(TypeError, pints.SumOfIndependentLogLikelihoods)
        self.assertRaises(
            ValueError, pints.SumOfIndependentLogLikelihoods, [l1])
        # Wrong types
        self.assertRaises(
            ValueError, pints.SumOfIndependentLogLikelihoods, [l1, 1])
        self.assertRaises(
            ValueError, pints.SumOfIndependentLogLikelihoods, [problem, l1])
        # Mismatching dimensions
        self.assertRaises(
            ValueError, pints.SumOfIndependentLogLikelihoods, [l1, l2])


if __name__ == '__main__':
    unittest.main()
