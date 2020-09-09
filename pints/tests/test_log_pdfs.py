#!/usr/bin/env python3
#
# Tests the log_posterior class
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import division
import unittest
import pints
import pints.toy
import numpy as np


class TestLogPosterior(unittest.TestCase):

    def test_log_posterior(self):

        # Create a toy problem and log likelihood
        model = pints.toy.LogisticModel()
        real_parameters = [0.015, 500]
        x = [0.014, 501]
        sigma = 0.001
        times = np.linspace(0, 1000, 100)
        values = model.simulate(real_parameters, times)
        problem = pints.SingleOutputProblem(model, times, values)
        log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, sigma)

        # Create a prior
        log_prior = pints.UniformLogPrior([0, 0], [1, 1000])

        # Test
        p = pints.LogPosterior(log_likelihood, log_prior)
        self.assertEqual(p(x), log_likelihood(x) + log_prior(x))
        y = [-1, 500]
        self.assertEqual(log_prior(y), -float('inf'))
        self.assertEqual(p(y), -float('inf'))
        self.assertEqual(p(y), log_prior(y))

        # Test derivatives
        log_prior = pints.ComposedLogPrior(
            pints.GaussianLogPrior(0.015, 0.3),
            pints.GaussianLogPrior(500, 100))
        log_posterior = pints.LogPosterior(log_likelihood, log_prior)
        x = [0.013, 540]
        y, dy = log_posterior.evaluateS1(x)
        self.assertEqual(y, log_posterior(x))
        self.assertEqual(dy.shape, (2, ))
        y1, dy1 = log_prior.evaluateS1(x)
        y2, dy2 = log_likelihood.evaluateS1(x)
        self.assertTrue(np.all(dy == dy1 + dy2))

        # Test getting the prior and likelihood back again
        self.assertIs(log_posterior.log_prior(), log_prior)
        self.assertIs(log_posterior.log_likelihood(), log_likelihood)

        # First arg must be a LogPDF
        self.assertRaises(ValueError, pints.LogPosterior, 'hello', log_prior)

        # Second arg must be a log_prior
        self.assertRaises(
            ValueError, pints.LogPosterior, log_likelihood, log_likelihood)

        # Prior and likelihood must have same dimension
        self.assertRaises(
            ValueError, pints.LogPosterior, log_likelihood,
            pints.GaussianLogPrior(0.015, 0.3))


class TestSumOfIndependentLogPDFs(unittest.TestCase):

    def test_sum_of_independent_log_pdfs(self):

        # Test single output
        model = pints.toy.LogisticModel()
        x = [0.015, 500]
        sigma = 0.1
        times = np.linspace(0, 1000, 100)
        values = model.simulate(x, times) + 0.1
        problem = pints.SingleOutputProblem(model, times, values)

        l1 = pints.GaussianKnownSigmaLogLikelihood(problem, sigma)
        l2 = pints.GaussianLogLikelihood(problem)
        ll = pints.SumOfIndependentLogPDFs([l1, l1, l1])
        self.assertEqual(l1.n_parameters(), ll.n_parameters())
        self.assertEqual(3 * l1(x), ll(x))

        # Test single output derivatives
        y, dy = ll.evaluateS1(x)
        self.assertEqual(y, ll(x))
        self.assertEqual(dy.shape, (2, ))
        y1, dy1 = l1.evaluateS1(x)
        self.assertTrue(np.all(3 * dy1 == dy))

        # Wrong number of arguments
        self.assertRaises(TypeError, pints.SumOfIndependentLogPDFs)
        self.assertRaises(
            ValueError, pints.SumOfIndependentLogPDFs, [l1])

        # Wrong types
        self.assertRaises(
            ValueError, pints.SumOfIndependentLogPDFs, [l1, 1])
        self.assertRaises(
            ValueError, pints.SumOfIndependentLogPDFs, [problem, l1])

        # Mismatching dimensions
        self.assertRaises(
            ValueError, pints.SumOfIndependentLogPDFs, [l1, l2])

        # Test multi-output
        model = pints.toy.FitzhughNagumoModel()
        x = model.suggested_parameters()
        nt = 10
        nx = model.n_parameters()
        times = np.linspace(0, 10, nt)
        values = model.simulate(x, times) + 0.01
        problem = pints.MultiOutputProblem(model, times, values)
        sigma = 0.01
        l1 = pints.GaussianKnownSigmaLogLikelihood(problem, sigma)
        ll = pints.SumOfIndependentLogPDFs([l1, l1, l1])
        self.assertEqual(l1.n_parameters(), ll.n_parameters())
        self.assertEqual(3 * l1(x), ll(x))

        # Test multi-output derivatives
        y, dy = ll.evaluateS1(x)

        # Note: y and ll(x) differ a bit, because the solver acts slightly
        # different when evaluating with and without sensitivities!
        self.assertAlmostEqual(y, ll(x), places=3)

        self.assertEqual(dy.shape, (nx, ))
        y1, dy1 = l1.evaluateS1(x)
        self.assertTrue(np.all(3 * dy1 == dy))



if __name__ == '__main__':
    unittest.main()
