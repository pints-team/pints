#!/usr/bin/env python3
#
# Tests the basic methods of the Hamiltonian MCMC routine.
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


class TestHamiltonianMCMC(unittest.TestCase):
    """
    Tests the basic methods of the Hamiltonian MCMC routine.
    """

    @classmethod
    def setUpClass(cls):
        """ Prepare a problem for testing. """

        # Create toy model
        cls.model = toy.LogisticModel()
        cls.real_parameters = np.array([0.015, 500])
        cls.times = np.linspace(0, 1000, 1000)
        cls.values = cls.model.simulate(cls.real_parameters, cls.times)

        # Add noise
        cls.noise = 10
        cls.values += np.random.normal(0, cls.noise, cls.values.shape)

        # Create an object with links to the model and time series
        cls.problem = pints.SingleOutputProblem(
            cls.model, cls.times, cls.values)

        # Create a uniform prior over the parameters
        cls.log_prior = pints.UniformLogPrior([0.01, 400], [0.02, 600])

        # Create a log likelihood
        cls.log_likelihood = pints.KnownNoiseLogLikelihood(
            cls.problem, cls.noise)

        # Create an un-normalised log-posterior (log-likelihood + log-prior)
        cls.log_posterior = pints.LogPosterior(
            cls.log_likelihood, cls.log_prior)

    def test_method(self):

        # Create mcmc
        x0 = self.real_parameters * 1.1
        mcmc = pints.HamiltonianMCMC(x0)

        # Set number of leapfrog steps
        ifrog = 10
        mcmc.set_leapfrog_steps(ifrog)

        # Perform short run
        chain = []
        for i in range(100 * ifrog):
            x = mcmc.ask()
            fx, gr = self.log_posterior.evaluateS1(x)
            sample = mcmc.tell((fx, gr))
            if i >= 50 * ifrog and sample is not None:
                chain.append(sample)
        chain = np.array(chain)
        self.assertEqual(chain.shape[0], 50)
        self.assertEqual(chain.shape[1], len(x0))

    def test_logging(self):
        """
        Test logging includes name and custom fields.
        """
        x = [self.real_parameters] * 3
        mcmc = pints.MCMCSampling(
            self.log_posterior, 3, x, method=pints.HamiltonianMCMC)
        mcmc.set_max_iterations(5)
        with StreamCapture() as c:
            mcmc.run()
        text = c.text()
        self.assertIn('Hamiltonian MCMC', text)
        self.assertIn(' iMCMC', text)
        self.assertIn(' iFrog', text)
        self.assertIn(' Accept.', text)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
