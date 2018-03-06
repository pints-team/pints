#!/usr/bin/env python3
#
# Tests the basic methods of the population MCMC routine.
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


class TestPopulationMCMC(unittest.TestCase):
    """
    Tests the basic methods of the population MCMC routine.
    """
    def __init__(self, name):
        super(TestPopulationMCMC, self).__init__(name)

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
        self.problem = pints.SingleSeriesProblem(
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
        x0 = self.real_parameters * 1.1
        mcmc = pints.PopulationMCMC(x0)

        # Test logging
        logger = pints.Logger()
        logger.set_stream(None)
        mcmc._log_init(logger)

        # Test schedule
        s = np.array([0, 0.1, 0.5])
        mcmc.set_schedule(s)
        self.assertTrue(np.all(s == mcmc.schedule()))

        # Perform short run
        chain = []
        for i in range(100):
            x = mcmc.ask()
            fx = self.log_posterior(x)
            sample = mcmc.tell(fx)
            if i >= 50:
                chain.append(sample)
            mcmc._log_write(logger)
        chain = np.array(chain)
        self.assertEqual(chain.shape[0], 50)
        self.assertEqual(chain.shape[1], len(x0))

        # Test name
        self.assertTrue('population' in mcmc.name().lower())

        #TODO: Add more stringent tests!

    def test_errors(self):

        mcmc = pints.PopulationMCMC(self.real_parameters)
        self.assertRaises(ValueError, mcmc.set_schedule, 1)
        self.assertRaises(ValueError, mcmc.set_schedule, [0])
        self.assertRaises(ValueError, mcmc.set_schedule, [0.5])
        mcmc.set_schedule([0, 0.5])
        self.assertRaises(ValueError, mcmc.set_schedule, [0.5, 0.5])
        self.assertRaises(ValueError, mcmc.set_schedule, [0, -0.1])
        self.assertRaises(ValueError, mcmc.set_schedule, [0, 1.1])

        mcmc = pints.PopulationMCMC(self.real_parameters)
        mcmc._initialise()
        self.assertRaises(RuntimeError, mcmc._initialise)
        self.assertRaises(RuntimeError, mcmc.set_schedule, [0, 0.1])

        mcmc = pints.PopulationMCMC(self.real_parameters)
        self.assertRaises(RuntimeError, mcmc.tell, 1)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
