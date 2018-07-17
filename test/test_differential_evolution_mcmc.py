#!/usr/bin/env python3
#
# Tests the basic methods of the differential evolution MCMC method.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import pints.toy as toy
import unittest
import numpy as np

debug = False


class TestDifferentialEvolutionMCMC(unittest.TestCase):
    """
    Tests the basic methods of the differential evolution MCMC method.
    """

    def __init__(self, name):
        super(TestDifferentialEvolutionMCMC, self).__init__(name)

        # Create toy model
        self.model = toy.LogisticModel()
        self.real_parameters = [0.015, 500]
        self.times = np.linspace(0, 1000, 1000)
        self.values = self.model.simulate(self.real_parameters, self.times)

        # Add noise
        self.noise = 10
        self.values += np.random.normal(0, self.noise, self.values.shape)
        self.real_parameters.append(self.noise)
        self.real_parameters = np.array(self.real_parameters)

        # Create an object with links to the model and time series
        self.problem = pints.SingleOutputProblem(
            self.model, self.times, self.values)

        # Create a uniform prior over both the parameters and the new noise
        # variable
        self.log_prior = pints.UniformLogPrior(
            [0.01, 400, self.noise * 0.1],
            [0.02, 600, self.noise * 100]
        )

        # Create a log likelihood
        self.log_likelihood = pints.UnknownNoiseLogLikelihood(self.problem)

        # Create an un-normalised log-posterior (log-likelihood + log-prior)
        self.log_posterior = pints.LogPosterior(
            self.log_likelihood, self.log_prior)

    def test_method(self):

        # Create mcmc
        xs = [
            self.real_parameters * 1.1,
            self.real_parameters * 1.05,
            self.real_parameters * 0.9,
            self.real_parameters * 0.95,
        ]
        mcmc = pints.DifferentialEvolutionMCMC(4, xs)

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
        # TODO: Add more stringent tests

    def test_flow(self):

        # Test we have at least 3 chains
        n = 2
        x0 = [self.real_parameters] * n
        self.assertRaises(ValueError, pints.DifferentialEvolutionMCMC, n, x0)

        # Test initial proposal is first point
        n = 3
        x0 = [self.real_parameters] * n
        mcmc = pints.DifferentialEvolutionMCMC(n, x0)
        self.assertTrue(mcmc.ask() is mcmc._x0)

        # Double initialisation
        mcmc = pints.DifferentialEvolutionMCMC(n, x0)
        mcmc.ask()
        self.assertRaises(RuntimeError, mcmc._initialise)

        # Tell without ask
        mcmc = pints.DifferentialEvolutionMCMC(n, x0)
        self.assertRaises(RuntimeError, mcmc.tell, 0)

        # Repeated asks should return same point
        mcmc = pints.DifferentialEvolutionMCMC(n, x0)
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
        mcmc = pints.DifferentialEvolutionMCMC(n, x0)
        mcmc.ask()
        self.assertRaises(ValueError, mcmc.tell, float('-inf'))

    def test_set_hyper_parameters(self):
        """
        Tests the hyper-parameter interface for this optimiser.
        """
        n = 3
        x0 = [self.real_parameters] * n
        mcmc = pints.DifferentialEvolutionMCMC(n, x0)

        self.assertEqual(mcmc.n_hyper_parameters(), 2)

        mcmc.set_hyper_parameters([0.5, 0.6])
        self.assertEqual(mcmc._gamma, 0.5)
        self.assertEqual(mcmc._b, 0.6)

        self.assertRaisesRegexp(
            ValueError, 'non-negative', mcmc.set_hyper_parameters, [-1, 0.5])

        self.assertRaisesRegexp(
            ValueError, 'non-negative', mcmc.set_hyper_parameters, [1, -0.5])


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
