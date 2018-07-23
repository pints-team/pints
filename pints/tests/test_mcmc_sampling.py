#!/usr/bin/env python2
#
# Tests the basic methods of the adaptive covariance MCMC routine.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import pints.toy
import unittest
import numpy as np

from shared import StreamCapture, TemporaryDirectory

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


debug = False

LOG_SCREEN = (
    'Using Adaptive covariance MCMC\n'
    'Generating 3 chains.\n'
    'Running in sequential mode.\n'
    'Iter. Eval. Accept.   Accept.   Accept.   Time m:s\n'
    '0     3      0         0         0          0:00.0\n'
    '1     6      0         0         0.5        0:00.0\n'
    '2     9      0         0         0.333      0:00.0\n'
    '3     12     0         0         0.5        0:00.0\n'
    '10    30     0.1       0         0.2        0:00.0\n'
    'Halting: Maximum number of iterations (10) reached.\n'
)

LOG_FILE = (
    'Iter. Eval. Accept.   Accept.   Accept.   Time m:s\n'
    '0     3      0         0         0          0:00.0\n'
    '1     6      0         0         0.5        0:00.0\n'
    '2     9      0         0         0.333      0:00.0\n'
    '3     12     0         0         0.5        0:00.0\n'
    '10    30     0.1       0         0.2        0:00.0\n'
)


class TestMCMCSampling(unittest.TestCase):
    """
    Tests the MCMCSampling class.
    """

    @classmethod
    def setUpClass(cls):
        """ Prepare problem for tests. """

        # Create toy model
        model = pints.toy.LogisticModel()
        cls.real_parameters = [0.015, 500]
        times = np.linspace(0, 1000, 1000)
        values = model.simulate(cls.real_parameters, times)

        # Add noise
        np.random.seed(1)
        cls.noise = 10
        values += np.random.normal(0, cls.noise, values.shape)
        cls.real_parameters.append(cls.noise)

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(model, times, values)

        # Create a uniform prior over both the parameters and the new noise
        # variable
        cls.log_prior = pints.UniformLogPrior(
            [0.01, 400, cls.noise * 0.1],
            [0.02, 600, cls.noise * 100]
        )

        # Create a log-likelihood
        cls.log_likelihood = pints.UnknownNoiseLogLikelihood(problem)

        # Create an un-normalised log-posterior (log-likelihood + log-prior)
        cls.log_posterior = pints.LogPosterior(
            cls.log_likelihood, cls.log_prior)

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
        mcmc.set_log_to_screen(False)
        self.assertEqual(len(mcmc.samplers()), nchains)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], nchains)
        self.assertEqual(chains.shape[1], niterations)
        self.assertEqual(chains.shape[2], nparameters)

        # Check constructor arguments
        pints.MCMCSampling(self.log_posterior, nchains, xs)
        pints.MCMCSampling(self.log_prior, nchains, xs)
        pints.MCMCSampling(self.log_likelihood, nchains, xs)

        def f(x):
            return x
        self.assertRaisesRegex(
            ValueError, 'extend pints.LogPDF', pints.MCMCSampling, f, nchains,
            xs)

        # Test x0 and chain argument
        self.assertRaisesRegex(
            ValueError, 'chains must be at least 1',
            pints.MCMCSampling, self.log_posterior, 0, [])
        self.assertRaisesRegex(
            ValueError, 'positions must be equal to number of chains',
            pints.MCMCSampling, self.log_posterior, 1, x0)
        self.assertRaisesRegex(
            ValueError, 'positions must be equal to number of chains',
            pints.MCMCSampling, self.log_posterior, 2, xs)
        self.assertRaisesRegex(
            ValueError, 'same dimension',
            pints.MCMCSampling, self.log_posterior, 1, [x0[:-1]])
        self.assertRaisesRegex(
            ValueError, 'extend pints.MCMCSampler',
            pints.MCMCSampling, self.log_posterior, 1, xs, method=12)

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
        mcmc.set_log_to_screen(False)
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
        nparameters = len(xs[0])
        niterations = 20
        mcmc = pints.MCMCSampling(
            self.log_posterior, nchains, xs,
            method=pints.AdaptiveCovarianceMCMC)
        mcmc.set_max_iterations(niterations)
        mcmc.set_log_to_screen(False)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], nchains)
        self.assertEqual(chains.shape[1], niterations)
        self.assertEqual(chains.shape[2], nparameters)

    def test_multi(self):

        # Set up problem for 10 chains
        x0 = np.array(self.real_parameters)
        xs = []
        for i in range(10):
            f = 0.9 + 0.2 * np.random.rand()
            xs.append(x0 * f)
        nchains = len(xs)
        nparameters = len(xs[0])
        niterations = 20

        # Test with multi-chain method
        meth = pints.DifferentialEvolutionMCMC
        mcmc = pints.MCMCSampling(
            self.log_posterior, nchains, xs, method=meth)
        self.assertEqual(len(mcmc.samplers()), 1)
        mcmc.set_max_iterations(niterations)
        mcmc.set_log_to_screen(False)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], nchains)
        self.assertEqual(chains.shape[1], niterations)
        self.assertEqual(chains.shape[2], nparameters)

        # Check constructor arguments
        pints.MCMCSampling(self.log_posterior, nchains, xs, method=meth)
        pints.MCMCSampling(self.log_prior, nchains, xs, method=meth)
        pints.MCMCSampling(self.log_likelihood, nchains, xs, method=meth)

        def f(x):
            return x
        self.assertRaisesRegex(
            ValueError, 'extend pints.LogPDF', pints.MCMCSampling, f, nchains,
            xs, method=meth)

        # Test x0 and chain argument
        self.assertRaisesRegex(
            ValueError, 'chains must be at least 1',
            pints.MCMCSampling, self.log_posterior, 0, [], method=meth)
        self.assertRaisesRegex(
            ValueError, 'positions must be equal to number of chains',
            pints.MCMCSampling, self.log_posterior, 1, x0, method=meth)
        self.assertRaisesRegex(
            ValueError, 'positions must be equal to number of chains',
            pints.MCMCSampling, self.log_posterior, 2, xs, method=meth)
        self.assertRaisesRegex(
            ValueError, 'same dimension',
            pints.MCMCSampling, self.log_posterior, 1, [x0[:-1]], method=meth)

        # Check different sigma0 initialisations
        pints.MCMCSampling(self.log_posterior, nchains, xs, method=meth)
        sigma0 = [0.005, 100, 0.5 * self.noise]
        pints.MCMCSampling(
            self.log_posterior, nchains, xs, sigma0, method=meth)
        sigma0 = np.diag([0.005, 100, 0.5 * self.noise])
        pints.MCMCSampling(
            self.log_posterior, nchains, xs, sigma0, method=meth)
        sigma0 = [0.005, 100, 0.5 * self.noise, 10]
        self.assertRaises(
            ValueError,
            pints.MCMCSampling, self.log_posterior, nchains, xs, sigma0,
            method=meth)
        sigma0 = np.diag([0.005, 100, 0.5 * self.noise, 10])
        self.assertRaises(
            ValueError,
            pints.MCMCSampling, self.log_posterior, nchains, xs, sigma0,
            method=meth)

    def test_stopping(self):
        """ Test different stopping criteria. """

        # Test without stopping criteria
        nchains = 1
        xs = [np.array(self.real_parameters) * 1.1]
        mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
        mcmc.set_max_iterations(None)
        self.assertRaises(ValueError, mcmc.run)

    def test_parallel(self):
        """ Test running MCMC with parallisation. """

        xs = []
        for i in range(10):
            f = 0.9 + 0.2 * np.random.rand()
            xs.append(np.array(self.real_parameters) * f)
        nchains = len(xs)
        nparameters = len(xs[0])
        niterations = 20
        mcmc = pints.MCMCSampling(
            self.log_posterior, nchains, xs,
            method=pints.AdaptiveCovarianceMCMC)
        mcmc.set_max_iterations(niterations)
        mcmc.set_log_to_screen(debug)

        # Test with auto-detected number of worker processes
        mcmc.set_parallel(True)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], nchains)
        self.assertEqual(chains.shape[1], niterations)
        self.assertEqual(chains.shape[2], nparameters)

        # Test with fixed number of worker processes
        mcmc.set_parallel(2)
        self.assertIs(mcmc._parallel, True)
        self.assertEqual(mcmc._n_workers, 2)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], nchains)
        self.assertEqual(chains.shape[1], niterations)
        self.assertEqual(chains.shape[2], nparameters)

    def test_logging(self):

        np.random.seed(1)
        xs = []
        for i in range(3):
            f = 0.9 + 0.2 * np.random.rand()
            xs.append(np.array(self.real_parameters) * f)
        nchains = len(xs)

        # No output
        with StreamCapture() as capture:
            mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
            mcmc.set_max_iterations(10)
            mcmc.set_log_to_screen(False)
            mcmc.run()
        self.assertEqual(capture.text(), '')

        # With output to screen
        np.random.seed(1)
        with StreamCapture() as capture:
            mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
            mcmc.set_max_iterations(10)
            mcmc.set_log_to_screen(True)
            mcmc.run()
        self.assertEqual(capture.text(), LOG_SCREEN)

        # With output to file
        np.random.seed(1)
        with StreamCapture() as capture:
            with TemporaryDirectory() as d:
                filename = d.path('test.txt')
                mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
                mcmc.set_max_iterations(10)
                mcmc.set_log_to_screen(False)
                mcmc.set_log_to_file(filename)
                mcmc.run()
                with open(filename, 'r') as f:
                    self.assertEqual(f.read(), LOG_FILE)
                    pass
            self.assertEqual(capture.text(), '')

        # Invalid log rate
        self.assertRaises(ValueError, mcmc.set_log_rate, 0)

    def test_adaptation(self):

        # 2 chains
        x0 = np.array(self.real_parameters) * 1.1
        x1 = np.array(self.real_parameters) * 1.15
        xs = [x0, x1]
        nchains = len(xs)

        # Delayed adaptation
        mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
        self.assertNotEqual(mcmc.adaptation_free_iterations(), 10)
        mcmc.set_adaptation_free_iterations(10)
        self.assertEqual(mcmc.adaptation_free_iterations(), 10)
        for sampler in mcmc._samplers:
            self.assertFalse(sampler.adaptation())
        mcmc.set_max_iterations(9)
        mcmc.set_log_to_screen(False)
        mcmc.run()
        for sampler in mcmc._samplers:
            self.assertFalse(sampler.adaptation())

        mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
        mcmc.set_adaptation_free_iterations(10)
        for sampler in mcmc._samplers:
            self.assertFalse(sampler.adaptation())
        mcmc.set_max_iterations(19)
        mcmc.set_log_to_screen(False)
        mcmc.run()
        for sampler in mcmc._samplers:
            self.assertTrue(sampler.adaptation())

        # No delay
        mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
        mcmc.set_adaptation_free_iterations(0)
        for sampler in mcmc._samplers:
            self.assertTrue(sampler.adaptation())
        mcmc.set_adaptation_free_iterations(0)
        for sampler in mcmc._samplers:
            self.assertTrue(sampler.adaptation())


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
