#!/usr/bin/env python
#
# Tests the basic methods of the Haario-Bardenet adaptive covariance MCMC
# routine.
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

from shared import StreamCapture

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestHaarioBardenetACMC(unittest.TestCase):
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

    def test_method(self):

        # Create mcmc
        x0 = self.real_parameters * 1.1
        mcmc = pints.HaarioBardenetACMC(x0)

        # Configure
        mcmc.set_target_acceptance_rate(0.3)
        mcmc.set_initial_phase(True)

        # Perform short run
        rate = []
        chain = []
        for i in range(100):
            x = mcmc.ask()
            fx = self.log_posterior(x)
            sample = mcmc.tell(fx)
            if i == 20:
                mcmc.set_initial_phase(False)
            if i >= 50:
                chain.append(sample)
            rate.append(mcmc.acceptance_rate())
            if np.all(sample == x):
                self.assertEqual(mcmc.current_log_pdf(), fx)

        chain = np.array(chain)
        rate = np.array(rate)
        self.assertEqual(chain.shape[0], 50)
        self.assertEqual(chain.shape[1], len(x0))
        self.assertEqual(rate.shape[0], 100)

    def test_hyperparameters(self):
        # Hyperparameters unchanged from base class
        mcmc = pints.HaarioBardenetACMC(self.real_parameters)
        self.assertEqual(mcmc.n_hyper_parameters(), 1)

    def test_name(self):
        # Test name method
        mcmc = pints.HaarioBardenetACMC(self.real_parameters)
        self.assertEqual(
            mcmc.name(), 'Haario-Bardenet adaptive covariance MCMC')

    def test_logging(self):
        # Test logging includes name.

        x = [self.real_parameters] * 3
        mcmc = pints.MCMCController(
            self.log_posterior, 3, x, method=pints.HaarioBardenetACMC)
        mcmc.set_max_iterations(5)
        with StreamCapture() as c:
            mcmc.run()
        text = c.text()
        self.assertIn('Haario-Bardenet adaptive covariance MCMC', text)

    def test_deprecated_alias(self):

        mcmc = pints.AdaptiveCovarianceMCMC(self.real_parameters)
        self.assertIn('Haario-Bardenet', mcmc.name())

        # Perform short run
        mcmc.set_target_acceptance_rate(0.3)
        mcmc.set_initial_phase(True)
        rate = []
        chain = []
        for i in range(100):
            x = mcmc.ask()
            fx = self.log_posterior(x)
            sample = mcmc.tell(fx)
            if i == 20:
                mcmc.set_initial_phase(False)
            if i >= 50:
                chain.append(sample)
            rate.append(mcmc.acceptance_rate())
            if np.all(sample == x):
                self.assertEqual(mcmc.current_log_pdf(), fx)
        chain = np.array(chain)
        rate = np.array(rate)
        self.assertEqual(chain.shape[0], 50)
        self.assertEqual(chain.shape[1], len(self.real_parameters))
        self.assertEqual(rate.shape[0], 100)


if __name__ == '__main__':
    unittest.main()
