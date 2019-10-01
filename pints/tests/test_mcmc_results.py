#!/usr/bin/env python
#
# Tests the basic methods of the adaptive covariance base class.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import pints.toy as toy
import unittest
import numpy as np
import time

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestAdaptiveCovarianceMC(unittest.TestCase):
    """
    Tests the basic methods of the adaptive covariance MCMC routine.
    """

    @classmethod
    def setUpClass(cls):
        """ Set up problem for tests. """

        # Create toy model
        cls.model = toy.LogisticModel()
        cls.real_parameters = [0.015, 500]
        cls.times = np.linspace(0, 1000, 1000)
        cls.values = cls.model.simulate(cls.real_parameters, cls.times)

        # Add noise
        cls.noise = 10
        cls.values += np.random.normal(0, cls.noise, cls.values.shape)
        cls.real_parameters.append(cls.noise)
        cls.real_parameters = np.array(cls.real_parameters)

        # Create an object with links to the model and time series
        cls.problem = pints.SingleOutputProblem(
            cls.model, cls.times, cls.values)

        # Create a uniform prior over both the parameters and the new noise
        # variable
        cls.log_prior = pints.UniformLogPrior(
            [0.01, 400, cls.noise * 0.1],
            [0.02, 600, cls.noise * 100]
        )

        # Create a log likelihood
        cls.log_likelihood = pints.GaussianLogLikelihood(cls.problem)

        # Create an un-normalised log-posterior (log-likelihood + log-prior)
        cls.log_posterior = pints.LogPosterior(
            cls.log_likelihood, cls.log_prior)

        # Run MCMC sampler
        xs = [cls.real_parameters * 1.1,
              cls.real_parameters * 0.9,
              cls.real_parameters * 1.15,
              ]

        mcmc = pints.MCMCController(cls.log_posterior, 3, xs,
                                    method=pints.HaarioBardenetACMC)
        mcmc.set_max_iterations(1000)
        mcmc.set_initial_phase_iterations(200)
        mcmc.set_log_to_screen(False)

        start = time.time()
        cls.chains = mcmc.run()
        end = time.time()
        cls.time = end - start

    def test_errors(self):
        # test errors occur when incorrectly calling MCMCResults
        self.assertRaises(ValueError, pints.MCMCResults, self.chains, -3)
        self.assertRaises(ValueError, pints.MCMCResults, self.chains, 0)
        self.assertRaises(ValueError, pints.MCMCResults, self.chains, 1.5,
                          ["param 1"])



if __name__ == '__main__':
    unittest.main()
