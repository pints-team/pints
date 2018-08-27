#!/usr/bin/env python
#
# Tests the basic methods of the DREAM MCMC method.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import unittest
import numpy as np

import pints
import pints.toy as toy

from shared import StreamCapture

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp

debug = False


class TestDreamMCMC(unittest.TestCase):
    """
    Tests the basic methods of the DREAM MCMC method.
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
        cls.log_likelihood = pints.UnknownNoiseLogLikelihood(cls.problem)

        # Create an un-normalised log-posterior (log-likelihood + log-prior)
        cls.log_posterior = pints.LogPosterior(
            cls.log_likelihood, cls.log_prior)

    def test_method(self):

        # Create mcmc
        xs = [
            self.real_parameters * 1.1,
            self.real_parameters * 1.05,
            self.real_parameters * 0.9,
            self.real_parameters * 0.95,
        ]
        mcmc = pints.DreamMCMC(4, xs)
        self.assertFalse(mcmc.constant_crossover())

        # Perform short run
        chains = []
        for i in range(100):
            xs = mcmc.ask()
            fxs = [self.log_posterior(x) for x in xs]
            samples = mcmc.tell(fxs)
            if i >= 50:
                chains.append(samples)
        chains = np.array(chains)
        self.assertEqual(chains.shape[0], 50)
        self.assertEqual(chains.shape[1], len(xs))
        self.assertEqual(chains.shape[2], len(xs[0]))

        # Repeat with constant crossover
        mcmc = pints.DreamMCMC(4, xs)
        mcmc.set_constant_crossover(True)
        self.assertTrue(mcmc.constant_crossover())

        # Perform short run
        chains = []
        for i in range(100):
            xs = mcmc.ask()
            fxs = [self.log_posterior(x) for x in xs]
            samples = mcmc.tell(fxs)
            if i >= 50:
                chains.append(samples)
        chains = np.array(chains)
        self.assertEqual(chains.shape[0], 50)
        self.assertEqual(chains.shape[1], len(xs))
        self.assertEqual(chains.shape[2], len(xs[0]))

        # Test getting/setting b
        b = mcmc.b()
        self.assertEqual(mcmc.b(), b)
        b += 0.01
        self.assertNotEqual(mcmc.b(), b)
        mcmc.set_b(b)
        self.assertEqual(mcmc.b(), b)

    def test_flow(self):

        # Test we have at least 3 chains
        n = 2
        x0 = [self.real_parameters] * n
        self.assertRaises(ValueError, pints.DreamMCMC, n, x0)

        # Test initial proposal is first point
        n = 3
        x0 = [self.real_parameters] * n
        mcmc = pints.DreamMCMC(n, x0)
        self.assertTrue(np.all(mcmc.ask() == mcmc._x0))

        # Double initialisation
        mcmc = pints.DreamMCMC(n, x0)
        mcmc.ask()
        self.assertRaises(RuntimeError, mcmc._initialise)

        # Tell without ask
        mcmc = pints.DreamMCMC(n, x0)
        self.assertRaises(RuntimeError, mcmc.tell, 0)

        # Repeated asks should return same point
        mcmc = pints.DreamMCMC(n, x0)
        # Get into accepting state
        for i in range(100):
            mcmc.tell([self.log_posterior(x) for x in mcmc.ask()])
        x = mcmc.ask()
        for i in range(10):
            self.assertTrue(x is mcmc.ask())

        # Repeated tells should fail
        mcmc.tell([1, 1, 1])
        self.assertRaises(RuntimeError, mcmc.tell, [1, 1, 1])

        # Bad starting point
        mcmc = pints.DreamMCMC(n, x0)
        mcmc.ask()
        self.assertRaises(ValueError, mcmc.tell, float('-inf'))

    def test_set_hyper_parameters(self):
        """
        Tests the hyper-parameter interface for this optimiser.
        """
        n = 3
        x0 = [self.real_parameters] * n
        mcmc = pints.DreamMCMC(n, x0)

        self.assertEqual(mcmc.n_hyper_parameters(), 1)

        mcmc.set_hyper_parameters([0.5])
        self.assertEqual(mcmc._b, 0.5)

    def test_logging(self):
        """
        Test logging includes name and custom fields.
        """
        x = [self.real_parameters] * 3
        mcmc = pints.MCMCSampling(
            self.log_posterior, 3, x, method=pints.DreamMCMC)
        mcmc.set_max_iterations(5)
        with StreamCapture() as c:
            mcmc.run()
        text = c.text()
        self.assertIn('DREAM', text)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
