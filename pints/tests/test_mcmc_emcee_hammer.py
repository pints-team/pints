#!/usr/bin/env python3
#
# Tests the basic functionality of the EmceeHammerMCMC method.
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


class TestEmceeHammerMCMC(unittest.TestCase):
    """
    Tests the basic functionality of the Emcee Hammer MCMC method.
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
        x0s = [
            self.real_parameters * 1.1,
            self.real_parameters * 1.05,
            self.real_parameters * 0.9,
            self.real_parameters * 0.95,
        ]
        mcmc = pints.EmceeHammerMCMC(4, x0s)

        # Perform short run
        chains = []
        for i in range(100):
            xs = mcmc.ask()
            fxs = np.array([self.log_posterior(x) for x in xs])
            ys, fys, ac = mcmc.tell(fxs)
            if i >= 50:
                chains.append(ys)
            # Note: not checking xs equal ys here

        chains = np.array(chains)
        self.assertEqual(chains.shape[0], 50)
        self.assertEqual(chains.shape[1], len(x0s))
        self.assertEqual(chains.shape[2], len(x0s[0]))

    def test_flow(self):

        # Test we have at least 3 chains
        n = 2
        x0 = [self.real_parameters] * n
        self.assertRaises(ValueError, pints.EmceeHammerMCMC, n, x0)

        # Test initial proposal is first point
        n = 3
        x0 = [self.real_parameters] * n
        mcmc = pints.EmceeHammerMCMC(n, x0)
        self.assertTrue(mcmc.ask() is mcmc._x0)

        # Double initialisation
        mcmc = pints.EmceeHammerMCMC(n, x0)
        mcmc.ask()
        self.assertRaises(RuntimeError, mcmc._initialise)

        # Tell without ask
        mcmc = pints.EmceeHammerMCMC(n, x0)
        self.assertRaises(RuntimeError, mcmc.tell, 0)

        # Repeated asks should return same point
        mcmc = pints.EmceeHammerMCMC(n, x0)
        # Get into accepting state
        for i in range(100):
            mcmc.tell([self.log_posterior(x) for x in mcmc.ask()])
        x = mcmc.ask()
        for i in range(10):
            self.assertTrue(x is mcmc.ask())

        # Repeated tells should fail
        mcmc.tell([1])
        self.assertRaises(RuntimeError, mcmc.tell, [1])

        # Bad starting point
        mcmc = pints.EmceeHammerMCMC(n, x0)
        mcmc.ask()
        self.assertRaises(ValueError, mcmc.tell, -np.inf)

    def test_set_hyper_parameters(self):
        # Tests the parameter and hyper-parameter interfaces for this sampler.

        n = 3
        x0 = [self.real_parameters] * n
        mcmc = pints.EmceeHammerMCMC(n, x0)

        self.assertEqual(mcmc.n_hyper_parameters(), 1)

        scale = mcmc.scale() + 0.1
        mcmc.set_hyper_parameters([scale])
        self.assertEqual(mcmc.scale(), scale)

        self.assertRaisesRegex(
            ValueError, 'exceed', mcmc.set_hyper_parameters, [1])

    def test_logging(self):
        # Test logging includes name and custom fields.

        x = [self.real_parameters] * 3
        mcmc = pints.MCMCController(
            self.log_posterior, 3, x, method=pints.EmceeHammerMCMC)
        mcmc.set_max_iterations(5)
        with StreamCapture() as c:
            mcmc.run()
        text = c.text()
        self.assertIn('Emcee Hammer MCMC', text)


if __name__ == '__main__':
    unittest.main()
