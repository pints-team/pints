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

from shared import StreamCapture

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestAdaptiveCovarianceMCMC(unittest.TestCase):
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

    def test_instantiation(self):

        # Create mcmc
        x0 = self.real_parameters * 1.1
        mcmc = pints.AdaptiveCovarianceMCMC(x0)
        self.assertEqual(0.6, mcmc.eta())
        self.assertEqual(0.234, mcmc.target_acceptance_rate())
        self.assertTrue(mcmc.in_initial_phase())

    def test_ask_tell(self):

        # ask only initialises
        x0 = self.real_parameters * 1.1
        mcmc = pints.AdaptiveCovarianceMCMC(x0)
        mcmc.ask()
        self.assertTrue(mcmc._running)

        # tell
        mcmc._proposed = None
        self.assertRaises(RuntimeError, mcmc.tell, 0.0)

    def test_replace(self):

        x0 = self.real_parameters * 1.1
        mcmc = pints.AdaptiveCovarianceMCMC(x0)

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
            ValueError, 'Dimension mismatch in `x`',
            mcmc.replace, [1, 2], 1)

        # Proposal can be changed too
        mcmc.ask()
        mcmc.replace([1, 2, 3], 10, [3, 4, 5])

        # New proposal must have correct size
        self.assertRaisesRegex(
            ValueError, '`proposed` has the wrong dimensions',
            mcmc.replace, [1, 2, 3], 3, [3, 4])

    def test_flow(self):

        # Test initial proposal is first point
        x0 = self.real_parameters
        mcmc = pints.AdaptiveCovarianceMCMC(x0)

        # Tell without ask
        mcmc = pints.AdaptiveCovarianceMCMC(x0)
        self.assertRaises(RuntimeError, mcmc.tell, 0)

        # Bad starting point
        mcmc = pints.AdaptiveCovarianceMCMC(x0)
        mcmc.ask()
        self.assertRaises(ValueError, mcmc.tell, float('-inf'))

    def test_options(self):

        # Test setting acceptance rate
        x0 = self.real_parameters
        mcmc = pints.AdaptiveCovarianceMCMC(x0)
        self.assertNotEqual(mcmc.target_acceptance_rate(), 0.5)
        mcmc.set_target_acceptance_rate(0.5)
        self.assertEqual(mcmc.target_acceptance_rate(), 0.5)
        mcmc.set_target_acceptance_rate(1)
        self.assertRaises(ValueError, mcmc.set_target_acceptance_rate, 0)
        self.assertRaises(ValueError, mcmc.set_target_acceptance_rate, -1e-6)
        self.assertRaises(ValueError, mcmc.set_target_acceptance_rate, 1.00001)
        mcmc.set_eta(0.3)
        self.assertEqual(mcmc.eta(), 0.3)

    def test_logging(self):
        """
        Test logging includes name and acceptance rate.
        """
        x = [self.real_parameters] * 3
        mcmc = pints.MCMCController(
            self.log_posterior, 3, x, method=pints.AdaptiveCovarianceMCMC)
        mcmc.set_max_iterations(5)
        with StreamCapture() as c:
            mcmc.run()
        text = c.text()
        self.assertIn('Adaptive covariance', text)
        self.assertIn('Accept.', text)


if __name__ == '__main__':
    unittest.main()
