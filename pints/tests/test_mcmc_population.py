#!/usr/bin/env python3
#
# Tests the basic methods of the population MCMC routine.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np

import pints
import pints.toy as toy

from shared import StreamCapture


class TestPopulationMCMC(unittest.TestCase):
    """
    Tests the basic methods of the population MCMC routine.
    """

    @classmethod
    def setUpClass(cls):
        """ Prepare a problem for testing. """

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
        mcmc = pints.PopulationMCMC(x0)

        # PopulationMCMC uses adaptive covariance internally, so requires an
        # initial phase
        self.assertTrue(mcmc.needs_initial_phase())
        self.assertTrue(mcmc.in_initial_phase())

        # Test schedule
        s = np.array([0, 0.1, 0.5])
        mcmc.set_temperature_schedule(s)
        self.assertTrue(np.all(s == mcmc.temperature_schedule()))

        # Perform short run
        chain = []
        for i in range(100):
            x = mcmc.ask()
            fx = self.log_posterior(x)
            y, fy, ac = mcmc.tell(fx)
            if i == 20:
                self.assertTrue(mcmc.in_initial_phase())
                mcmc.set_initial_phase(False)
                self.assertFalse(mcmc.in_initial_phase())
            if i >= 50:
                chain.append(y)
            self.assertEqual(fy, self.log_posterior(y))

        chain = np.array(chain)
        self.assertEqual(chain.shape[0], 50)
        self.assertEqual(chain.shape[1], len(x0))

        #TODO: Add more stringent tests!

    def test_errors(self):

        mcmc = pints.PopulationMCMC(self.real_parameters)
        self.assertRaises(ValueError, mcmc.set_temperature_schedule, 1)
        self.assertRaises(ValueError, mcmc.set_temperature_schedule, [0])
        self.assertRaises(ValueError, mcmc.set_temperature_schedule, [0.5])
        mcmc.set_temperature_schedule([0, 0.5])
        self.assertRaises(
            ValueError, mcmc.set_temperature_schedule, [0.5, 0.5])
        self.assertRaises(ValueError, mcmc.set_temperature_schedule, [0, -0.1])
        self.assertRaises(ValueError, mcmc.set_temperature_schedule, [0, 1.1])

        mcmc = pints.PopulationMCMC(self.real_parameters)
        mcmc._initialise()
        self.assertRaises(RuntimeError, mcmc._initialise)
        self.assertRaises(
            RuntimeError, mcmc.set_temperature_schedule, [0, 0.1])

        mcmc = pints.PopulationMCMC(self.real_parameters)
        self.assertRaises(RuntimeError, mcmc.tell, 1)

    def test_logging(self):
        # Test logging includes name and custom fields.

        x = [self.real_parameters] * 3
        mcmc = pints.MCMCController(
            self.log_posterior, 3, x, method=pints.PopulationMCMC)
        mcmc.set_max_iterations(5)
        with StreamCapture() as c:
            mcmc.run()
        text = c.text()
        self.assertIn('Population MCMC', text)
        self.assertIn(' i    ', text)
        self.assertIn(' j    ', text)
        self.assertIn(' Ex. ', text)

    def test_hyperparameters(self):

        mcmc = pints.PopulationMCMC(self.real_parameters)
        self.assertEqual(mcmc.n_hyper_parameters(), 1)

        # Test setting with an int
        mcmc.set_temperature_schedule(7)
        x = mcmc.temperature_schedule()
        self.assertEqual(len(x), 7)
        mcmc.set_temperature_schedule(8)
        self.assertEqual(len(mcmc.temperature_schedule()), 8)
        mcmc.set_hyper_parameters([7])
        y = mcmc.temperature_schedule()
        self.assertEqual(len(y), 7)
        self.assertTrue(np.all(x == y))

        # Test setting with a list
        mcmc.set_temperature_schedule(x)
        self.assertRaisesRegex(
            ValueError, 'First hyper-parameter',
            mcmc.set_hyper_parameters, [x])


if __name__ == '__main__':
    unittest.main()
