#!/usr/bin/env python3
#
# Tests the log_posterior class
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import division
import unittest
import pints
import pints.toy
import numpy as np


class Testlog_posterior(unittest.TestCase):

    def test_log_posterior(self):

        # Create a toy problem and log likelihood
        model = pints.toy.LogisticModel()
        real_parameters = [0.015, 500]
        x = [0.014, 501]
        sigma = 0.001
        times = np.linspace(0, 1000, 100)
        values = model.simulate(real_parameters, times)
        problem = pints.SingleOutputProblem(model, times, values)
        log_likelihood = pints.KnownNoiseLogLikelihood(problem, sigma)

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
            pints.NormalLogPrior(0.015, 0.3),
            pints.NormalLogPrior(500, 100))
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
            pints.NormalLogPrior(0.015, 0.3))


if __name__ == '__main__':
    unittest.main()
