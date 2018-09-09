#!/usr/bin/env python3
#
# Tests the basic methods of the population MCMC routine.
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

debug = False


class TestPopulationMCMC(unittest.TestCase):
    """
    Tests the basic methods of the Emcee Hammer MCMC routine.
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
        num_chains = 10
        xs = [self.real_parameters * (1 + 0.1 * np.random.rand()) for i in range(num_chains)]
        mcmc = pints.EmceeHammerMCMC(xs)

        # Perform short run
        chain = []
        for i in range(100):
            x = mcmc.ask()
            fx = self.log_posterior(x)
            sample = mcmc.tell(fx)
            chain.append(sample)
        chain = np.array(chain)
        self.assertEqual(chain.shape[0], 100)
        self.assertEqual(chain.shape[1], len(x0))

    def test_errors(self):

        mcmc = pints.PopulationMCMC(self.real_parameters)
        self.assertRaises(ValueError, mcmc.set_a, -1)
        self.assertEqual(mcmc.n_hyper_parameters(), 1)
        self.assertRaises(ValueError, mcmc.set_hyper_parameters, [0])
        mcmc.set_hyper_parameters([1])
        self.assertEqual(mcmc._a, 1)

    def test_logging(self):
        """
        Test logging includes name and custom fields.
        """
        x = [self.real_parameters] * 3
        mcmc = pints.MCMCSampling(
            self.log_posterior, 3, x, method=pints.EmceeHammerMCMC)
        mcmc.set_max_iterations(5)
        with StreamCapture() as c:
            mcmc.run()
        text = c.text()
        self.assertIn('Emcee Hammer MCMC', text)
        self.assertIn(' i    ', text)
        self.assertIn(' j    ', text)
        self.assertIn(' Ex. ', text)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
