#!/usr/bin/env python3
#
# Tests the basic methods of the adaptive covariance base class.
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

    def test_ask_tell(self):

        # Test ask-tell flow

        # Test initial proposal is first point
        x0 = self.real_parameters
        mcmc = pints.HaarioACMC(x0)
        self.assertTrue(np.all(x0 == mcmc.ask()))

        # Repeated calls return same initial point
        self.assertTrue(np.all(x0 == mcmc.ask()))
        self.assertTrue(np.all(x0 == mcmc.ask()))

        # Repeated asks should return same point
        mcmc = pints.HaarioACMC(x0)
        # Get into accepting state
        mcmc.set_initial_phase(False)
        for i in range(100):
            mcmc.tell(self.log_posterior(mcmc.ask()))
        x = mcmc.ask()
        for i in range(10):
            self.assertTrue(x is mcmc.ask())

        # Repeated tells should fail
        mcmc.tell(1)
        self.assertRaises(RuntimeError, mcmc.tell, 1)

        # Bad starting point
        mcmc = pints.HaarioACMC(x0)
        mcmc.ask()
        self.assertRaises(ValueError, mcmc.tell, -np.inf)

        # Tell without ask
        mcmc = pints.HaarioACMC(x0)
        self.assertRaises(RuntimeError, mcmc.tell, 0)

        # Bad starting point
        mcmc = pints.HaarioACMC(x0)
        mcmc.ask()
        self.assertRaises(ValueError, mcmc.tell, -np.inf)

    def test_eta(self):
        # Test eta getting and setting

        mcmc = pints.HaarioACMC(self.real_parameters)
        self.assertEqual(mcmc.eta(), 0.6)
        mcmc.set_eta(0.1)
        self.assertEqual(mcmc.eta(), 0.1)
        mcmc.set_eta(0.4)
        self.assertEqual(mcmc.eta(), 0.4)

        self.assertRaisesRegex(
            ValueError, 'greater than zero', mcmc.set_eta, 0)
        self.assertRaisesRegex(
            ValueError, 'greater than zero', mcmc.set_eta, -0.1)

    def test_hyper_parameters(self):
        # Tests hyperparameter methods

        mcmc = pints.HaarioACMC(self.real_parameters)
        self.assertTrue(mcmc.n_hyper_parameters(), 1)
        mcmc.set_hyper_parameters([0.1])
        self.assertEqual(mcmc.eta(), 0.1)
        mcmc.set_hyper_parameters([0.67])
        self.assertEqual(mcmc.eta(), 0.67)

    def test_initial_phase(self):
        # Test initial phase setting

        mcmc = pints.HaarioACMC(self.real_parameters)
        self.assertTrue(mcmc.needs_initial_phase())
        self.assertTrue(mcmc.in_initial_phase())
        mcmc.set_initial_phase(True)
        self.assertTrue(mcmc.in_initial_phase())
        mcmc.set_initial_phase(False)
        self.assertFalse(mcmc.in_initial_phase())
        mcmc.set_initial_phase(True)
        self.assertTrue(mcmc.in_initial_phase())

    def test_logging(self):
        # Test logging includes acceptance rate, evaluations, iterations and
        # time.

        x = [self.real_parameters] * 3
        mcmc = pints.MCMCController(
            self.log_posterior, 3, x, method=pints.HaarioACMC)
        mcmc.set_max_iterations(5)
        with StreamCapture() as c:
            mcmc.run()
        text = c.text()
        self.assertIn('Accept.', text)
        self.assertIn('Eval.', text)
        self.assertIn('Iter.', text)
        self.assertIn('Time m:s', text)

    def test_replace(self):
        # Tests the replace() method

        x0 = self.real_parameters
        mcmc = pints.HaarioACMC(x0)

        # One round of ask-tell must have been run
        self.assertRaisesRegex(
            RuntimeError, 'already running', mcmc.replace, x0, 1)

        mcmc.ask()

        # One round of ask-tell must have been run
        self.assertRaises(RuntimeError, mcmc.replace, x0, 1)

        mcmc.tell(0.5)
        mcmc.replace([1, 2, 3], 10)
        mcmc.replace([1, 2, 3], 10)

        # New position must have correct size
        self.assertRaisesRegex(
            ValueError, 'Point `current` has the wrong dimensions',
            mcmc.replace, [1, 2], 1)

        # Proposal can be changed too
        mcmc.ask()
        mcmc.replace([1, 2, 3], 10, [3, 4, 5])

        # New proposal must have correct size
        self.assertRaisesRegex(
            ValueError, '`proposed` has the wrong dimensions',
            mcmc.replace, [1, 2, 3], 3, [3, 4])

    def test_target_acceptance_rate(self):
        # Test target_acceptance_rate getting and setting

        mcmc = pints.HaarioACMC(self.real_parameters)
        self.assertEqual(mcmc.target_acceptance_rate(), 0.234)

        mcmc.set_target_acceptance_rate(0.1)
        self.assertEqual(mcmc.target_acceptance_rate(), 0.1)

        self.assertRaises(ValueError, mcmc.set_target_acceptance_rate, 0)
        self.assertRaises(ValueError, mcmc.set_target_acceptance_rate, -1e-6)
        self.assertRaises(ValueError, mcmc.set_target_acceptance_rate, 1.00001)


if __name__ == '__main__':
    unittest.main()
