#!/usr/bin/env python3
#
# Tests the LogPosterior class
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
        loglikelihood = pints.KnownNoiseLogLikelihood(problem, sigma)

        # Create a prior
        logprior = pints.UniformLogPrior([0, 0], [1, 1000])

        # Test
        p = pints.LogPosterior(loglikelihood, logprior)
        self.assertEqual(p(x), loglikelihood(x) + logprior(x))
        y = [-1, 500]
        self.assertEqual(logprior(y), -float('inf'))
        self.assertEqual(p(y), -float('inf'))
        self.assertEqual(p(y), logprior(y))


if __name__ == '__main__':
    unittest.main()
