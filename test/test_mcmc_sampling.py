#!/usr/bin/env python3
#
# Tests the basic methods of the adaptive covariance MCMC routine.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import pints.io
import pints.toy
import unittest
import numpy as np

debug = False


class TestMCMCSampling(unittest.TestCase):
    """
    Tests the MCMCSampling class.
    """
    def __init__(self, name):
        super(TestMCMCSampling, self).__init__(name)

        # Create toy model
        self.model = pints.toy.LogisticModel()
        self.real_parameters = [0.015, 500]
        self.times = np.linspace(0, 1000, 1000)
        self.values = self.model.simulate(self.real_parameters, self.times)

        # Add noise
        self.noise = 10
        self.values += np.random.normal(0, self.noise, self.values.shape)
        self.real_parameters.append(self.noise)

        # Create an object with links to the model and time series
        self.problem = pints.SingleSeriesProblem(
            self.model, self.times, self.values)

        # Create a uniform prior over both the parameters and the new noise
        # variable
        self.log_prior = pints.UniformLogPrior(
            [0.01, 400, self.noise * 0.1],
            [0.02, 600, self.noise * 100]
        )

        # Create a log-likelihood
        self.log_likelihood = pints.UnknownNoiseLogLikelihood(self.problem)

        # Create an un-normalised log-posterior (log-likelihood + log-prior)
        self.log_posterior = pints.LogPosterior(
            self.log_likelihood, self.log_prior)

    def test_single(self):
        """ Test with a SingleChainMCMC method. """

        # One chain
        nchains = 1

        # Test simple run
        x0 = np.array(self.real_parameters) * 1.1
        xs = [x0]
        nparameters = len(x0)
        niterations = 10
        mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
        mcmc.set_max_iterations(niterations)
        mcmc.set_verbose(False)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], nchains)
        self.assertEqual(chains.shape[1], niterations)
        self.assertEqual(chains.shape[2], nparameters)

        # Check function argument
        pints.MCMCSampling(self.log_posterior, nchains, xs)
        pints.MCMCSampling(self.log_prior, nchains, xs)
        pints.MCMCSampling(self.log_likelihood, nchains, xs)

        def f(x):
            return x
        self.assertRaises(ValueError, pints.MCMCSampling, f, nchains, xs)

        # Test x0 and chain argument
        self.assertRaises(
            ValueError, pints.MCMCSampling, self.log_posterior, 0, [])
        self.assertRaises(
            ValueError, pints.MCMCSampling, self.log_posterior, 1, x0)
        self.assertRaises(
            ValueError, pints.MCMCSampling, self.log_posterior, 2, xs)

        # Check different sigma0 initialisations
        pints.MCMCSampling(self.log_posterior, nchains, xs)
        sigma0 = [0.005, 100, 0.5 * self.noise]
        pints.MCMCSampling(self.log_posterior, nchains, xs, sigma0)
        sigma0 = np.diag([0.005, 100, 0.5 * self.noise])
        pints.MCMCSampling(self.log_posterior, nchains, xs, sigma0)
        sigma0 = [0.005, 100, 0.5 * self.noise, 10]
        self.assertRaises(
            ValueError,
            pints.MCMCSampling, self.log_posterior, nchains, xs, sigma0)
        sigma0 = np.diag([0.005, 100, 0.5 * self.noise, 10])
        self.assertRaises(
            ValueError,
            pints.MCMCSampling, self.log_posterior, nchains, xs, sigma0)

        # Test multi-chain with single-chain mcmc

        # 2 chains
        x0 = np.array(self.real_parameters) * 1.1
        x1 = np.array(self.real_parameters) * 1.15
        xs = [x0, x1]
        nchains = len(xs)
        nparameters = len(x0)
        niterations = 10
        mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
        mcmc.set_max_iterations(niterations)
        mcmc.set_verbose(False)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], nchains)
        self.assertEqual(chains.shape[1], niterations)
        self.assertEqual(chains.shape[2], nparameters)

        # 10 chains
        xs = []
        for i in range(10):
            f = 0.9 + 0.2 * np.random.rand()
            xs.append(np.array(self.real_parameters) * f)
        nchains = len(xs)
        nparameters = len(x0)
        niterations = 20
        mcmc = pints.MCMCSampling(
            self.log_posterior, nchains, xs,
            method=pints.AdaptiveCovarianceMCMC)
        mcmc.set_max_iterations(niterations)
        mcmc.set_verbose(False)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], nchains)
        self.assertEqual(chains.shape[1], niterations)
        self.assertEqual(chains.shape[2], nparameters)

        # Test with multi-chain method
        mcmc = pints.MCMCSampling(
            self.log_posterior, nchains, xs,
            method=pints.DifferentialEvolutionMCMC)
        mcmc.set_max_iterations(niterations)
        mcmc.set_verbose(False)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], nchains)
        self.assertEqual(chains.shape[1], niterations)
        self.assertEqual(chains.shape[2], nparameters)

        # Test verbose switch
        with pints.io.StdOutCapture() as capture:
            mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
            mcmc.set_max_iterations(niterations)
            mcmc.set_verbose(False)
            chains = mcmc.run()
        self.assertEqual(capture.text(), '')
        with pints.io.StdOutCapture() as capture:
            mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
            mcmc.set_max_iterations(niterations)
            mcmc.set_verbose(True)
            chains = mcmc.run()
        self.assertNotEqual(capture.text(), '')

        # Test without stopping criteria
        mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
        mcmc.set_max_iterations(None)
        self.assertRaises(ValueError, mcmc.run)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
