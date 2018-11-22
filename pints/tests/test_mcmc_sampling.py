#!/usr/bin/env python3
#
# Tests the basic methods of the adaptive covariance MCMC routine.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import os
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

LOG_SCREEN = [
    'Using Adaptive covariance MCMC',
    'Generating 3 chains.',
    'Running in sequential mode.',
    'Iter. Eval. Accept.   Accept.   Accept.   Time m:s',
    '0     3      0         0         0          0:00.0',
    '1     6      0         0         0.5        0:00.0',
    '2     9      0         0         0.333      0:00.0',
    '3     12     0         0         0.5        0:00.0',
    'Initial phase completed.',
    '10    30     0.1       0.1       0.2        0:00.0',
    'Halting: Maximum number of iterations (10) reached.',
]

LOG_FILE = [
    'Iter. Eval. Accept.   Accept.   Accept.   Time m:s',
    '0     3      0         0         0          0:00.0',
    '1     6      0         0         0.5        0:00.0',
    '2     9      0         0         0.333      0:00.0',
    '3     12     0         0         0.5        0:00.0',
    '10    30     0.1       0.1       0.2        0:00.0',
]


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

        # Check sampler() method
        self.assertRaises(RuntimeError, mcmc.sampler)

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

        # Test x0 and chain argument
        self.assertRaisesRegex(
            ValueError, 'chains must be at least 1', meth, 0, [])
        self.assertRaisesRegex(
            ValueError, 'at least 3',
            meth, 1, [x0])
        self.assertRaisesRegex(
            ValueError, 'positions must be equal to number of chains',
            meth, 5, xs)
        self.assertRaisesRegex(
            ValueError, 'same dimension',
            meth, 3, [x0, x0, x0[:-1]])

        # Check sampler() method
        self.assertIsInstance(mcmc.sampler(), pints.MultiChainMCMC)

        # Check different sigma0 initialisations work
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

        nchains = 1
        xs = [np.array(self.real_parameters) * 1.1]
        mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)

        # Test setting max iterations
        maxi = mcmc.max_iterations() + 2
        self.assertNotEqual(maxi, mcmc.max_iterations())
        mcmc.set_max_iterations(maxi)
        self.assertEqual(maxi, mcmc.max_iterations())
        self.assertRaisesRegex(
            ValueError, 'negative', mcmc.set_max_iterations, -1)

        # Test without stopping criteria
        mcmc.set_max_iterations(None)
        self.assertIsNone(mcmc.max_iterations())
        self.assertRaisesRegex(
            ValueError, 'At least one stopping criterion', mcmc.run)

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
        self.assertFalse(mcmc.parallel())
        mcmc.set_parallel(True)
        self.assertTrue(mcmc.parallel())
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], nchains)
        self.assertEqual(chains.shape[1], niterations)
        self.assertEqual(chains.shape[2], nparameters)

        # Test with fixed number of worker processes
        mcmc.set_parallel(2)
        mcmc.set_log_to_screen(True)
        self.assertIs(mcmc._parallel, True)
        self.assertEqual(mcmc._n_workers, 2)
        with StreamCapture() as c:
            chains = mcmc.run()
        self.assertIn('with 2 worker', c.text())
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
            mcmc.set_initial_phase_iterations(5)
            mcmc.set_max_iterations(10)
            mcmc.set_log_to_screen(False)
            mcmc.set_log_to_file(False)
            mcmc.run()
        self.assertEqual(capture.text(), '')

        # With output to screen
        np.random.seed(1)
        with StreamCapture() as capture:
            mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
            mcmc.set_initial_phase_iterations(5)
            mcmc.set_max_iterations(10)
            mcmc.set_log_to_screen(True)
            mcmc.set_log_to_file(False)
            mcmc.run()
        lines = capture.text().splitlines()
        for i, line in enumerate(lines):
            self.assertLess(i, len(LOG_SCREEN))
            self.assertEqual(line, LOG_SCREEN[i])
        self.assertEqual(len(lines), len(LOG_SCREEN))

        # With output to file
        np.random.seed(1)
        with StreamCapture() as capture:
            with TemporaryDirectory() as d:
                filename = d.path('test.txt')
                mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
                mcmc.set_initial_phase_iterations(5)
                mcmc.set_max_iterations(10)
                mcmc.set_log_to_screen(False)
                mcmc.set_log_to_file(filename)
                mcmc.run()
                with open(filename, 'r') as f:
                    lines = f.read().splitlines()
                for i, line in enumerate(lines):
                    self.assertLess(i, len(LOG_FILE))
                    self.assertEqual(line, LOG_FILE[i])
                self.assertEqual(len(lines), len(LOG_FILE))
            self.assertEqual(capture.text(), '')

        # Invalid log interval
        self.assertRaises(ValueError, mcmc.set_log_interval, 0)

    def test_initial_phase(self):

        # 2 chains
        x0 = np.array(self.real_parameters) * 1.1
        x1 = np.array(self.real_parameters) * 1.15
        xs = [x0, x1]
        nchains = len(xs)

        # Initial phase
        mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
        self.assertTrue(mcmc.method_needs_initial_phase())
        self.assertNotEqual(mcmc.initial_phase_iterations(), 10)
        mcmc.set_initial_phase_iterations(10)
        self.assertEqual(mcmc.initial_phase_iterations(), 10)
        self.assertRaisesRegex(
            ValueError, 'negative', mcmc.set_initial_phase_iterations, -1)
        for sampler in mcmc._samplers:
            self.assertTrue(sampler.in_initial_phase())
        mcmc.set_max_iterations(9)
        mcmc.set_log_to_screen(False)
        mcmc.run()
        for sampler in mcmc._samplers:
            self.assertTrue(sampler.in_initial_phase())

        mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
        mcmc.set_initial_phase_iterations(10)
        for sampler in mcmc._samplers:
            self.assertTrue(sampler.in_initial_phase())
        mcmc.set_max_iterations(11)
        mcmc.set_log_to_screen(False)
        mcmc.run()
        for sampler in mcmc._samplers:
            self.assertFalse(sampler.in_initial_phase())

        # No initial phase
        mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
        mcmc.set_initial_phase_iterations(0)
        mcmc.set_max_iterations(1)
        mcmc.set_log_to_screen(False)
        mcmc.run()
        for sampler in mcmc._samplers:
            self.assertFalse(sampler.in_initial_phase())

    def test_live_chain_and_eval_logging(self):

        np.random.seed(1)
        xs = []
        for i in range(3):
            f = 0.9 + 0.2 * np.random.rand()
            xs.append(np.array(self.real_parameters) * f)
        nchains = len(xs)

        # Test writing chains - not evals to disk (using LogPosterior)
        mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
        mcmc.set_initial_phase_iterations(5)
        mcmc.set_max_iterations(20)
        mcmc.set_log_to_screen(True)
        mcmc.set_log_to_file(False)

        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                cpath = d.path('chain.csv')
                p0 = d.path('chain_0.csv')
                p1 = d.path('chain_1.csv')
                p2 = d.path('chain_2.csv')
                epath = d.path('evals.csv')
                p3 = d.path('evals_0.csv')
                p4 = d.path('evals_1.csv')
                p5 = d.path('evals_2.csv')

                # Test files aren't created before mcmc runs
                mcmc.set_chain_filename(cpath)
                mcmc.set_log_pdf_filename(None)
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertFalse(os.path.exists(p0))
                self.assertFalse(os.path.exists(p1))
                self.assertFalse(os.path.exists(p2))
                self.assertFalse(os.path.exists(p3))
                self.assertFalse(os.path.exists(p4))
                self.assertFalse(os.path.exists(p5))

                # Test files are created afterwards
                chains1 = mcmc.run()
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertTrue(os.path.exists(p0))
                self.assertTrue(os.path.exists(p1))
                self.assertTrue(os.path.exists(p2))
                self.assertFalse(os.path.exists(p3))
                self.assertFalse(os.path.exists(p4))
                self.assertFalse(os.path.exists(p5))

                # Test files contain the correct chains
                import pints.io as io
                chains2 = np.array(io.load_samples(cpath, nchains))
                self.assertTrue(np.all(chains1 == chains2))

            text = c.text()
            self.assertIn('Writing chains to', text)
            self.assertIn('chain_0.csv', text)
            self.assertNotIn('Writing evaluations to', text)
            self.assertNotIn('evals_0.csv', text)

        # Test writing evals - not chains to disk (using LogPosterior)
        mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
        mcmc.set_initial_phase_iterations(5)
        mcmc.set_max_iterations(20)
        mcmc.set_log_to_screen(True)
        mcmc.set_log_to_file(False)

        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                cpath = d.path('chain.csv')
                p0 = d.path('chain_0.csv')
                p1 = d.path('chain_1.csv')
                p2 = d.path('chain_2.csv')
                epath = d.path('evals.csv')
                p3 = d.path('evals_0.csv')
                p4 = d.path('evals_1.csv')
                p5 = d.path('evals_2.csv')

                # Test files aren't created before mcmc runs
                mcmc.set_chain_filename(None)
                mcmc.set_log_pdf_filename(epath)
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertFalse(os.path.exists(p0))
                self.assertFalse(os.path.exists(p1))
                self.assertFalse(os.path.exists(p2))
                self.assertFalse(os.path.exists(p3))
                self.assertFalse(os.path.exists(p4))
                self.assertFalse(os.path.exists(p5))

                # Test files are created afterwards
                chains1 = mcmc.run()
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertFalse(os.path.exists(p0))
                self.assertFalse(os.path.exists(p1))
                self.assertFalse(os.path.exists(p2))
                self.assertTrue(os.path.exists(p3))
                self.assertTrue(os.path.exists(p4))
                self.assertTrue(os.path.exists(p5))

                # Test files contain the correct values
                import pints.io as io
                evals2 = np.array(io.load_samples(epath, nchains))
                evals1 = []
                for chain in chains1:
                    logpdfs = np.array([self.log_posterior(x) for x in chain])
                    logpriors = np.array([self.log_prior(x) for x in chain])
                    loglikelihoods = logpdfs - logpriors
                    evals = np.array([logpdfs, loglikelihoods, logpriors]).T
                    evals1.append(evals)
                evals1 = np.array(evals1)
                self.assertTrue(np.all(evals1 == evals2))

            text = c.text()
            self.assertNotIn('Writing chains to', text)
            self.assertNotIn('chain_0.csv', text)
            self.assertIn('Writing evaluations to', text)
            self.assertIn('evals_0.csv', text)

        # Test writing chains and evals to disk (with LogPosterior)
        mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
        mcmc.set_initial_phase_iterations(5)
        mcmc.set_max_iterations(20)
        mcmc.set_log_to_screen(True)
        mcmc.set_log_to_file(False)

        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                cpath = d.path('chain.csv')
                p0 = d.path('chain_0.csv')
                p1 = d.path('chain_1.csv')
                p2 = d.path('chain_2.csv')
                epath = d.path('evals.csv')
                p3 = d.path('evals_0.csv')
                p4 = d.path('evals_1.csv')
                p5 = d.path('evals_2.csv')

                # Test files aren't created before mcmc runs
                mcmc.set_chain_filename(cpath)
                mcmc.set_log_pdf_filename(epath)
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertFalse(os.path.exists(p0))
                self.assertFalse(os.path.exists(p1))
                self.assertFalse(os.path.exists(p2))
                self.assertFalse(os.path.exists(p3))
                self.assertFalse(os.path.exists(p4))
                self.assertFalse(os.path.exists(p5))

                # Test files are created afterwards
                chains1 = mcmc.run()
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertTrue(os.path.exists(p0))
                self.assertTrue(os.path.exists(p1))
                self.assertTrue(os.path.exists(p2))
                self.assertTrue(os.path.exists(p3))
                self.assertTrue(os.path.exists(p4))
                self.assertTrue(os.path.exists(p5))

                # Test chain files contain the correct values
                import pints.io as io
                chains2 = np.array(io.load_samples(cpath, nchains))
                self.assertTrue(np.all(chains1 == chains2))

                # Test eval files contain the correct values
                evals2 = np.array(io.load_samples(epath, nchains))
                evals1 = []
                for chain in chains1:
                    logpdfs = np.array([self.log_posterior(x) for x in chain])
                    logpriors = np.array([self.log_prior(x) for x in chain])
                    loglikelihoods = logpdfs - logpriors
                    evals = np.array([logpdfs, loglikelihoods, logpriors]).T
                    evals1.append(evals)
                evals1 = np.array(evals1)
                self.assertTrue(np.all(evals1 == evals2))

            text = c.text()
            self.assertIn('Writing chains to', text)
            self.assertIn('chain_0.csv', text)
            self.assertIn('Writing evaluations to', text)
            self.assertIn('evals_0.csv', text)

        # Test writing chains and evals to disk (with LogLikelihood)
        mcmc = pints.MCMCSampling(self.log_likelihood, nchains, xs)
        mcmc.set_initial_phase_iterations(5)
        mcmc.set_max_iterations(20)
        mcmc.set_log_to_screen(True)
        mcmc.set_log_to_file(False)

        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                cpath = d.path('chain.csv')
                p0 = d.path('chain_0.csv')
                p1 = d.path('chain_1.csv')
                p2 = d.path('chain_2.csv')
                epath = d.path('evals.csv')
                p3 = d.path('evals_0.csv')
                p4 = d.path('evals_1.csv')
                p5 = d.path('evals_2.csv')

                # Test files aren't created before mcmc runs
                mcmc.set_chain_filename(cpath)
                mcmc.set_log_pdf_filename(epath)
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertFalse(os.path.exists(p0))
                self.assertFalse(os.path.exists(p1))
                self.assertFalse(os.path.exists(p2))
                self.assertFalse(os.path.exists(p3))
                self.assertFalse(os.path.exists(p4))
                self.assertFalse(os.path.exists(p5))

                # Test files are created afterwards
                chains1 = mcmc.run()
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertTrue(os.path.exists(p0))
                self.assertTrue(os.path.exists(p1))
                self.assertTrue(os.path.exists(p2))
                self.assertTrue(os.path.exists(p3))
                self.assertTrue(os.path.exists(p4))
                self.assertTrue(os.path.exists(p5))

                # Test chain files contain the correct values
                import pints.io as io
                chains2 = np.array(io.load_samples(cpath, nchains))
                self.assertTrue(np.all(chains1 == chains2))

                # Test eval files contain the correct values
                evals2 = np.array(io.load_samples(epath, nchains))
                evals1 = []
                for chain in chains1:
                    evals1.append(
                        np.array([self.log_likelihood(x) for x in chain]).T)
                evals1 = np.array(evals1).reshape(3, 20, 1)
                self.assertTrue(np.all(evals1 == evals2))

            text = c.text()
            self.assertIn('Writing chains to', text)
            self.assertIn('chain_0.csv', text)
            self.assertIn('Writing evaluations to', text)
            self.assertIn('evals_0.csv', text)

        # Test logging can be disabled again
        mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
        mcmc.set_initial_phase_iterations(5)
        mcmc.set_max_iterations(20)
        mcmc.set_log_to_screen(True)
        mcmc.set_log_to_file(False)

        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                cpath = d.path('chain.csv')
                p0 = d.path('chain_0.csv')
                p1 = d.path('chain_1.csv')
                p2 = d.path('chain_2.csv')
                epath = d.path('evals.csv')
                p3 = d.path('evals_0.csv')
                p4 = d.path('evals_1.csv')
                p5 = d.path('evals_2.csv')

                # Test files aren't created before mcmc runs
                mcmc.set_chain_filename(cpath)
                mcmc.set_log_pdf_filename(epath)
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertFalse(os.path.exists(p0))
                self.assertFalse(os.path.exists(p1))
                self.assertFalse(os.path.exists(p2))
                self.assertFalse(os.path.exists(p3))
                self.assertFalse(os.path.exists(p4))
                self.assertFalse(os.path.exists(p5))

                # Test files are not created afterwards
                mcmc.set_chain_filename(None)
                mcmc.set_log_pdf_filename(None)
                mcmc.run()
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertFalse(os.path.exists(p0))
                self.assertFalse(os.path.exists(p1))
                self.assertFalse(os.path.exists(p2))
                self.assertFalse(os.path.exists(p3))
                self.assertFalse(os.path.exists(p4))
                self.assertFalse(os.path.exists(p5))

            text = c.text()
            self.assertNotIn('Writing chains to', text)
            self.assertNotIn('chain_0.csv', text)
            self.assertNotIn('Writing evaluations to', text)
            self.assertNotIn('evals_0.csv', text)

        # Test with a single chain
        nchains = 1
        mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs[:1])
        mcmc.set_initial_phase_iterations(5)
        mcmc.set_max_iterations(20)
        mcmc.set_log_to_screen(True)
        mcmc.set_log_to_file(False)

        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                cpath = d.path('chain.csv')
                p0 = d.path('chain_0.csv')
                p1 = d.path('chain_1.csv')
                p2 = d.path('chain_2.csv')
                epath = d.path('evals.csv')
                p3 = d.path('evals_0.csv')
                p4 = d.path('evals_1.csv')
                p5 = d.path('evals_2.csv')

                # Test files aren't created before mcmc runs
                mcmc.set_chain_filename(cpath)
                mcmc.set_log_pdf_filename(epath)
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertFalse(os.path.exists(p0))
                self.assertFalse(os.path.exists(p1))
                self.assertFalse(os.path.exists(p2))
                self.assertFalse(os.path.exists(p3))
                self.assertFalse(os.path.exists(p4))
                self.assertFalse(os.path.exists(p5))

                # Test files are created afterwards
                chains1 = mcmc.run()
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertTrue(os.path.exists(p0))
                self.assertFalse(os.path.exists(p1))
                self.assertFalse(os.path.exists(p2))
                self.assertTrue(os.path.exists(p3))
                self.assertFalse(os.path.exists(p4))
                self.assertFalse(os.path.exists(p5))

                # Test chain files contain the correct values
                import pints.io as io
                chains2 = np.array(io.load_samples(cpath, nchains))
                self.assertTrue(np.all(chains1 == chains2))

                # Test eval files contain the correct values
                evals2 = np.array(io.load_samples(epath, nchains))
                evals1 = []
                for chain in chains1:
                    logpdfs = np.array([self.log_posterior(x) for x in chain])
                    logpriors = np.array([self.log_prior(x) for x in chain])
                    loglikelihoods = logpdfs - logpriors
                    evals = np.array([logpdfs, loglikelihoods, logpriors]).T
                    evals1.append(evals)
                evals1 = np.array(evals1)
                self.assertTrue(np.all(evals1 == evals2))

            text = c.text()
            self.assertIn('Writing chains to', text)
            self.assertIn('chain_0.csv', text)
            self.assertIn('Writing evaluations to', text)
            self.assertIn('evals_0.csv', text)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
