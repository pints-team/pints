#!/usr/bin/env python3
#
# Tests the basic methods of the Haario-Bardenet adaptive covariance MCMC
# routine.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.toy as toy
import unittest
import numpy as np

from shared import StreamCapture


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
            y, fy, ac = mcmc.tell(fx)
            if i == 20:
                mcmc.set_initial_phase(False)
            if i >= 50:
                chain.append(x)
            rate.append(mcmc.acceptance_rate())
            self.assertTrue(isinstance(ac, bool))
            if ac:
                self.assertTrue(np.all(x == y))
                self.assertEqual(fx, fy)

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
            y, fy, ac = mcmc.tell(fx)
            if i == 20:
                mcmc.set_initial_phase(False)
            if i >= 50:
                chain.append(x)
            rate.append(mcmc.acceptance_rate())
            self.assertTrue(isinstance(ac, bool))
            if ac:
                self.assertTrue(np.all(x == y))
                self.assertEqual(fx, fy)
        chain = np.array(chain)
        rate = np.array(rate)
        self.assertEqual(chain.shape[0], 50)
        self.assertEqual(chain.shape[1], len(self.real_parameters))
        self.assertEqual(rate.shape[0], 100)


if __name__ == '__main__':
    unittest.main()
