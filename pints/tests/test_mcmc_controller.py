#!/usr/bin/env python3
#
# Tests the MCMC Controller.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import os
import pints
import pints.io
import pints.toy
import unittest
import unittest.mock
import numpy as np
import numpy.testing as npt
import warnings

from shared import StreamCapture, TemporaryDirectory


debug = False

LOG_SCREEN = [
    'Using Haario-Bardenet adaptive covariance MCMC',
    'Generating 3 chains.',
    'Running in sequential mode.',
    'Iter. Eval. Accept.   Accept.   Accept.   Time    ',
    '0     3      0         0         0          0:00.0',
    '1     6      0         0         0.5        0:00.0',
    '2     9      0         0         0.333      0:00.0',
    '3     12     0         0         0.5        0:00.0',
    'Initial phase completed.',
    '10    30     0.1       0.1       0.2        0:00.0',
    'Halting: Maximum number of iterations (10) reached.',
]

LOG_FILE = [
    'Iter. Eval. Accept.   Accept.   Accept.   Time    ',
    '0     3      0         0         0          0:00.0',
    '1     6      0         0         0.5        0:00.0',
    '2     9      0         0         0.333      0:00.0',
    '3     12     0         0         0.5        0:00.0',
    '10    30     0.1       0.1       0.2        0:00.0',
]


class TestMCMCController(unittest.TestCase):
    """
    Tests the MCMCController class.
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
        cls.log_likelihood = pints.GaussianLogLikelihood(problem)

        # Create an un-normalised log-posterior (log-likelihood + log-prior)
        cls.log_posterior = pints.LogPosterior(
            cls.log_likelihood, cls.log_prior)

        # Create another log-likelihood with two noise parameters
        cls.log_likelihood_2 = pints.AR1LogLikelihood(problem)
        cls.log_prior_2 = pints.UniformLogPrior(
            [0.01, 400, 0.0, 0.0],
            [0.02, 600, 100, 1]
        )

        # Create an un-normalised log-posterior (log-likelihood + log-prior)
        cls.log_posterior_2 = pints.LogPosterior(
            cls.log_likelihood_2, cls.log_prior_2)

    def test_single(self):
        # Test with a SingleChainMCMC method.

        # One chain
        n_chains = 1

        # Test simple run
        x0 = np.array(self.real_parameters) * 1.1
        xs = [x0]
        n_parameters = len(x0)
        n_iterations = 10
        mcmc = pints.MCMCController(self.log_posterior, n_chains, xs)
        mcmc.set_max_iterations(n_iterations)
        mcmc.set_log_to_screen(False)
        self.assertEqual(len(mcmc.samplers()), n_chains)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], n_chains)
        self.assertEqual(chains.shape[1], n_iterations)
        self.assertEqual(chains.shape[2], n_parameters)

        # Test chains() method
        self.assertIs(chains, mcmc.chains())

        # Check constructor arguments
        pints.MCMCController(self.log_posterior, n_chains, xs)
        pints.MCMCController(self.log_prior, n_chains, xs)
        pints.MCMCController(self.log_likelihood, n_chains, xs)

        # Check sampler() method
        self.assertRaises(RuntimeError, mcmc.sampler)

        def f(x):
            return x
        self.assertRaisesRegex(
            TypeError, 'extend pints.LogPDF', pints.MCMCController,
            f, n_chains, xs)

        # Test x0 and chain argument
        self.assertRaisesRegex(
            ValueError, 'chains must be at least 1',
            pints.MCMCController, self.log_posterior, 0, [])
        self.assertRaisesRegex(
            ValueError, 'positions must be equal to number of chains',
            pints.MCMCController, self.log_posterior, 1, x0)
        self.assertRaisesRegex(
            ValueError, 'positions must be equal to number of chains',
            pints.MCMCController, self.log_posterior, 2, xs)
        self.assertRaisesRegex(
            ValueError, 'same dimension',
            pints.MCMCController, self.log_posterior, 1, [x0[:-1]])
        self.assertRaisesRegex(
            ValueError, 'extend pints.MCMCSampler',
            pints.MCMCController, self.log_posterior, 1, xs, method=12)

        # Check different sigma0 initialisations
        pints.MCMCController(self.log_posterior, n_chains, xs)
        sigma0 = [0.005, 100, 0.5 * self.noise]
        pints.MCMCController(self.log_posterior, n_chains, xs, sigma0)
        sigma0 = np.diag([0.005, 100, 0.5 * self.noise])
        pints.MCMCController(self.log_posterior, n_chains, xs, sigma0)
        sigma0 = [0.005, 100, 0.5 * self.noise, 10]
        self.assertRaises(
            ValueError,
            pints.MCMCController, self.log_posterior, n_chains, xs, sigma0)
        sigma0 = np.diag([0.005, 100, 0.5 * self.noise, 10])
        self.assertRaises(
            ValueError,
            pints.MCMCController, self.log_posterior, n_chains, xs, sigma0)

        # Test transformation
        logt = pints.LogTransformation(n_parameters)
        mcmc = pints.MCMCController(self.log_posterior, n_chains, xs,
                                    transformation=logt)
        mcmc.set_max_iterations(n_iterations)
        mcmc.set_log_to_screen(False)
        chains = mcmc.run()
        # Test chains inverse transformed
        # log-transformation of the parameter in [0.01, 0.02] will always be
        # negative values, so checking it larger than zero make sure it's
        # transformed back to the model space.
        self.assertTrue(np.all(chains > 0))
        self.assertEqual(chains.shape[0], n_chains)
        self.assertEqual(chains.shape[1], n_iterations)
        self.assertEqual(chains.shape[2], n_parameters)
        sigma0 = [0.005, 100, 0.5 * self.noise]
        pints.MCMCController(self.log_posterior, n_chains, xs, sigma0,
                             transformation=logt)
        sigma0 = np.diag([0.005, 100, 0.5 * self.noise])
        pints.MCMCController(self.log_posterior, n_chains, xs, sigma0,
                             transformation=logt)
        sigma0 = [0.005, 100, 0.5 * self.noise, 10]
        self.assertRaises(
            ValueError,
            pints.MCMCController, self.log_posterior, n_chains, xs, sigma0,
            transformation=logt)
        sigma0 = np.diag([0.005, 100, 0.5 * self.noise, 10])
        self.assertRaises(
            ValueError,
            pints.MCMCController, self.log_posterior, n_chains, xs, sigma0,
            transformation=logt)
        sigma0 = np.arange(16).reshape(2, 2, 2, 2)
        self.assertRaises(
            ValueError,
            pints.MCMCController, self.log_posterior, n_chains, xs, sigma0,
            transformation=logt)

        # Test multi-chain with single-chain mcmc

        # 2 chains
        x0 = np.array(self.real_parameters) * 1.1
        x1 = np.array(self.real_parameters) * 1.15
        xs = [x0, x1]
        n_chains = len(xs)
        n_parameters = len(x0)
        n_iterations = 10
        mcmc = pints.MCMCController(self.log_posterior, n_chains, xs)
        mcmc.set_max_iterations(n_iterations)
        mcmc.set_log_to_screen(False)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], n_chains)
        self.assertEqual(chains.shape[1], n_iterations)
        self.assertEqual(chains.shape[2], n_parameters)
        self.assertIs(chains, mcmc.chains())

        # 10 chains
        xs = []
        for i in range(10):
            f = 0.9 + 0.2 * np.random.rand()
            xs.append(np.array(self.real_parameters) * f)
        n_chains = len(xs)
        n_parameters = len(xs[0])
        n_iterations = 20
        mcmc = pints.MCMCController(
            self.log_posterior, n_chains, xs,
            method=pints.HaarioBardenetACMC)
        mcmc.set_max_iterations(n_iterations)
        mcmc.set_log_to_screen(False)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], n_chains)
        self.assertEqual(chains.shape[1], n_iterations)
        self.assertEqual(chains.shape[2], n_parameters)
        self.assertIs(chains, mcmc.chains())

    def test_hyperparameters_constant(self):
        # Test that sampler hyperparameter remain same before and after run

        # single chain method
        n_chains = 1
        x0 = np.array(self.real_parameters) * 1.1
        xs = [x0]
        mcmc = pints.MCMCController(
            self.log_posterior, n_chains, xs, method=pints.HamiltonianMCMC)
        step_size = 0.77
        for sampler in mcmc.samplers():
            sampler.set_leapfrog_step_size(step_size)
        mcmc.set_max_iterations(5)
        mcmc.set_log_to_screen(False)
        mcmc.run()
        for sampler in mcmc.samplers():
            self.assertEqual(sampler.leapfrog_step_size()[0], step_size)

        # test multiple chain method
        # Set up problem for 10 chains
        x0 = np.array(self.real_parameters)
        xs = []
        for i in range(10):
            f = 0.9 + 0.2 * np.random.rand()
            xs.append(x0 * f)
        n_chains = len(xs)

        meth = pints.DifferentialEvolutionMCMC
        mcmc = pints.MCMCController(
            self.log_posterior, n_chains, xs, method=meth)
        switch_rate = 4
        mcmc.samplers()[0].set_gamma_switch_rate(switch_rate)
        mcmc.set_max_iterations(5)
        mcmc.set_log_to_screen(False)
        mcmc.run()
        self.assertEqual(mcmc.samplers()[0].gamma_switch_rate(), switch_rate)

    def test_multi(self):
        # Test with a multi-chain method

        # Set up problem for 10 chains
        x0 = np.array(self.real_parameters)
        xs = []
        for i in range(10):
            f = 0.9 + 0.2 * np.random.rand()
            xs.append(x0 * f)
        n_chains = len(xs)
        n_parameters = len(xs[0])
        n_iterations = 20

        # Test with multi-chain method
        meth = pints.DifferentialEvolutionMCMC
        mcmc = pints.MCMCController(
            self.log_posterior, n_chains, xs, method=meth)
        self.assertEqual(len(mcmc.samplers()), 1)
        mcmc.set_max_iterations(n_iterations)
        mcmc.set_log_to_screen(False)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], n_chains)
        self.assertEqual(chains.shape[1], n_iterations)
        self.assertEqual(chains.shape[2], n_parameters)

        # Test chains() method
        self.assertIs(chains, mcmc.chains())

        # Check constructor arguments
        pints.MCMCController(self.log_posterior, n_chains, xs, method=meth)
        pints.MCMCController(self.log_prior, n_chains, xs, method=meth)
        pints.MCMCController(self.log_likelihood, n_chains, xs, method=meth)

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
        pints.MCMCController(self.log_posterior, n_chains, xs, method=meth)
        sigma0 = [0.005, 100, 0.5 * self.noise]
        pints.MCMCController(
            self.log_posterior, n_chains, xs, sigma0, method=meth)
        sigma0 = np.diag([0.005, 100, 0.5 * self.noise])
        pints.MCMCController(
            self.log_posterior, n_chains, xs, sigma0, method=meth)
        sigma0 = [0.005, 100, 0.5 * self.noise, 10]
        self.assertRaises(
            ValueError,
            pints.MCMCController, self.log_posterior, n_chains, xs, sigma0,
            method=meth)
        sigma0 = np.diag([0.005, 100, 0.5 * self.noise, 10])
        self.assertRaises(
            ValueError,
            pints.MCMCController, self.log_posterior, n_chains, xs, sigma0,
            method=meth)

        # Test transformation
        logt = pints.LogTransformation(n_parameters)
        mcmc = pints.MCMCController(self.log_posterior, n_chains, xs,
                                    method=meth, transformation=logt)
        self.assertEqual(len(mcmc.samplers()), 1)
        mcmc.set_max_iterations(n_iterations)
        mcmc.set_log_to_screen(False)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], n_chains)
        self.assertEqual(chains.shape[1], n_iterations)
        self.assertEqual(chains.shape[2], n_parameters)
        # Test chains inverse transformed
        # log-transformation of the parameter in [0.01, 0.02] will always be
        # negative values, so checking it larger than zero make sure it's
        # transformed back to the model space.
        self.assertTrue(np.all(chains > 0))
        sigma0 = [0.005, 100, 0.5 * self.noise]
        pints.MCMCController(self.log_posterior, n_chains, xs, sigma0,
                             method=meth, transformation=logt)
        sigma0 = np.diag([0.005, 100, 0.5 * self.noise])
        pints.MCMCController(self.log_posterior, n_chains, xs, sigma0,
                             method=meth, transformation=logt)
        sigma0 = [0.005, 100, 0.5 * self.noise, 10]
        self.assertRaises(
            ValueError,
            pints.MCMCController, self.log_posterior, n_chains, xs, sigma0,
            method=meth, transformation=logt)
        sigma0 = np.diag([0.005, 100, 0.5 * self.noise, 10])
        self.assertRaises(
            ValueError,
            pints.MCMCController, self.log_posterior, n_chains, xs, sigma0,
            method=meth, transformation=logt)

    def test_multi_logpdf(self):
        # Test with multiple logpdfs

        # 2 chains
        x0 = np.array(self.real_parameters) * 1.1
        x1 = np.array(self.real_parameters) * 1.15
        xs = [x0, x1]

        # Not iterable
        with self.assertRaises(TypeError):
            mcmc = pints.MCMCController(1, 3, xs)

        # Wrong number of logpdfs
        with self.assertRaises(ValueError):
            mcmc = pints.MCMCController(
                [self.log_posterior, self.log_posterior], 3, xs)

        # List does not contain logpdfs
        with self.assertRaises(ValueError):
            mcmc = pints.MCMCController(
                [self.log_posterior, 'abc'], 2, xs)

        # Pdfs have different numbers of n_parameters
        with self.assertRaises(ValueError):
            mcmc = pints.MCMCController(
                [self.log_posterior, self.log_posterior_2], 2, xs)

        # Correctly configured inputs
        n_chains = len(xs)
        n_parameters = len(x0)
        n_iterations = 10
        mcmc = pints.MCMCController(
            [self.log_posterior, self.log_posterior],
            n_chains,
            xs,
            transformation=pints.LogTransformation(n_parameters),
            sigma0=[1, 0.1, 0.01])
        mcmc.set_max_iterations(n_iterations)
        mcmc.set_log_to_screen(False)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], n_chains)
        self.assertEqual(chains.shape[1], n_iterations)
        self.assertEqual(chains.shape[2], n_parameters)
        self.assertIs(chains, mcmc.chains())

        # With sensitivities needed
        mcmc = pints.MCMCController(
            [self.log_posterior, self.log_posterior],
            n_chains,
            xs,
            transformation=pints.LogTransformation(n_parameters),
            sigma0=[1, 0.1, 0.01],
            method=pints.HamiltonianMCMC)
        mcmc.set_max_iterations(n_iterations)
        mcmc.set_log_to_screen(False)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], n_chains)
        self.assertEqual(chains.shape[1], n_iterations)
        self.assertEqual(chains.shape[2], n_parameters)
        self.assertIs(chains, mcmc.chains())

        # Parallel (currently raises error)
        mcmc = pints.MCMCController(
            [self.log_posterior, self.log_posterior],
            n_chains,
            xs,
            transformation=pints.LogTransformation(n_parameters),
            sigma0=[1, 0.1, 0.01])
        mcmc.set_parallel(True)
        mcmc.set_max_iterations(n_iterations)
        mcmc.set_log_to_screen(False)
        with self.assertRaises(ValueError):
            chains = mcmc.run()

        # Test that both logpdfs are called
        logpdf1 = unittest.mock.MagicMock(
            return_value=-1.0, spec=self.log_posterior)
        logpdf2 = unittest.mock.MagicMock(
            return_value=-2.0, spec=self.log_posterior)
        attrs = {'n_parameters.return_value': 3}
        logpdf1.configure_mock(**attrs)
        logpdf2.configure_mock(**attrs)
        mcmc = pints.MCMCController([logpdf1, logpdf2], n_chains, xs)
        mcmc.set_max_iterations(n_iterations)
        mcmc.set_log_to_screen(False)
        chains = mcmc.run()

        logpdf1.assert_called()
        logpdf2.assert_called()

        # Check that they got called with the corresponding x0 at the start
        npt.assert_allclose(logpdf1.call_args_list[0][0][0], xs[0])
        npt.assert_allclose(logpdf2.call_args_list[0][0][0], xs[1])

    def test_stopping(self):
        # Test different stopping criteria.

        nchains = 1
        xs = [np.array(self.real_parameters) * 1.1]
        mcmc = pints.MCMCController(self.log_posterior, nchains, xs)

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
        # Test running MCMC with parallisation.

        xs = []
        for i in range(10):
            f = 0.9 + 0.2 * np.random.rand()
            xs.append(np.array(self.real_parameters) * f)
        nchains = len(xs)
        nparameters = len(xs[0])
        niterations = 20
        mcmc = pints.MCMCController(
            self.log_posterior, nchains, xs,
            method=pints.HaarioBardenetACMC)
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
        mcmc = pints.MCMCController(
            self.log_posterior, nchains, xs,
            method=pints.HaarioBardenetACMC)
        mcmc.set_max_iterations(niterations)
        mcmc.set_log_to_screen(debug)
        mcmc.set_parallel(5)
        mcmc.set_log_to_screen(True)
        self.assertIs(mcmc._parallel, True)
        self.assertEqual(mcmc._n_workers, 5)
        with StreamCapture() as c:
            chains = mcmc.run()
        self.assertIn('with 5 worker', c.text())
        self.assertEqual(chains.shape[0], nchains)
        self.assertEqual(chains.shape[1], niterations)
        self.assertEqual(chains.shape[2], nparameters)

    def test_logging(self):
        # Test logging functions

        np.random.seed(1)
        xs = []
        for i in range(3):
            f = 0.9 + 0.2 * np.random.rand()
            xs.append(np.array(self.real_parameters) * f)
        nchains = len(xs)

        # No output
        with StreamCapture() as capture:
            mcmc = pints.MCMCController(self.log_posterior, nchains, xs)
            mcmc.set_initial_phase_iterations(5)
            mcmc.set_max_iterations(10)
            mcmc.set_log_to_screen(False)
            mcmc.set_log_to_file(False)
            mcmc.run()
        self.assertEqual(capture.text(), '')

        # With output to screen
        np.random.seed(1)
        with StreamCapture() as capture:
            mcmc = pints.MCMCController(self.log_posterior, nchains, xs)
            mcmc.set_initial_phase_iterations(5)
            mcmc.set_max_iterations(10)
            mcmc.set_log_to_screen(True)
            mcmc.set_log_to_file(False)
            mcmc.run()
        lines = capture.text().splitlines()
        for i, line in enumerate(lines):
            self.assertLess(i, len(LOG_SCREEN))
            # Chop off time bit before comparison
            if LOG_SCREEN[i][-6:] == '0:00.0':
                self.assertEqual(line[:-6], LOG_SCREEN[i][:-6])
            else:
                self.assertEqual(line, LOG_SCREEN[i])
        self.assertEqual(len(lines), len(LOG_SCREEN))

        # With output to file
        np.random.seed(1)
        with StreamCapture() as capture:
            with TemporaryDirectory() as d:
                filename = d.path('test.txt')
                mcmc = pints.MCMCController(self.log_posterior, nchains, xs)
                mcmc.set_initial_phase_iterations(5)
                mcmc.set_max_iterations(10)
                mcmc.set_log_to_screen(False)
                mcmc.set_log_to_file(filename)
                mcmc.run()
                with open(filename, 'r') as f:
                    lines = f.read().splitlines()
                for i, line in enumerate(lines):
                    self.assertLess(i, len(LOG_FILE))
                    # Chop off time bit before comparison
                    if LOG_FILE[i][-6:] == '0:00.0':
                        self.assertEqual(line[:-6], LOG_FILE[i][:-6])
                    else:
                        self.assertEqual(line, LOG_FILE[i])
                    self.assertEqual(line[:-6], LOG_FILE[i][:-6])
                self.assertEqual(len(lines), len(LOG_FILE))
            self.assertEqual(capture.text(), '')

        # Invalid log interval
        self.assertRaises(ValueError, mcmc.set_log_interval, 0)

    def test_initial_phase(self):
        # Test if the initial phase functions work

        # 2 chains
        x0 = np.array(self.real_parameters) * 1.1
        x1 = np.array(self.real_parameters) * 1.15
        xs = [x0, x1]
        nchains = len(xs)

        # Initial phase
        mcmc = pints.MCMCController(self.log_posterior, nchains, xs)
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

        mcmc = pints.MCMCController(self.log_posterior, nchains, xs)
        mcmc.set_initial_phase_iterations(10)
        for sampler in mcmc._samplers:
            self.assertTrue(sampler.in_initial_phase())
        mcmc.set_max_iterations(11)
        mcmc.set_log_to_screen(False)
        mcmc.run()
        for sampler in mcmc._samplers:
            self.assertFalse(sampler.in_initial_phase())

        # No initial phase
        mcmc = pints.MCMCController(self.log_posterior, nchains, xs)
        mcmc.set_initial_phase_iterations(0)
        mcmc.set_max_iterations(1)
        mcmc.set_log_to_screen(False)
        mcmc.run()
        for sampler in mcmc._samplers:
            self.assertFalse(sampler.in_initial_phase())

    def test_log_pdf_storage_in_memory_single(self):
        # Test storing evaluations in memory, with a single-chain method

        # Set up test problem
        x0 = np.array(self.real_parameters) * 1.05
        x1 = np.array(self.real_parameters) * 1.15
        x2 = np.array(self.real_parameters) * 0.95
        xs = [x0, x1, x2]
        n_chains = len(xs)
        n_iterations = 100

        # Single-chain method, using a logposterior
        mcmc = pints.MCMCController(self.log_posterior, n_chains, xs)
        mcmc.set_max_iterations(n_iterations)
        mcmc.set_log_to_screen(False)
        mcmc.set_log_pdf_storage(True)
        chains = mcmc.run()

        # Test shape of returned array
        evals = mcmc.log_pdfs()
        self.assertEqual(len(evals.shape), 3)
        self.assertEqual(evals.shape[0], n_chains)
        self.assertEqual(evals.shape[1], n_iterations)
        self.assertEqual(evals.shape[2], 3)

        # Test returned values
        for i, chain in enumerate(chains):
            posteriors = [self.log_posterior(x) for x in chain]
            self.assertTrue(np.all(evals[i, :, 0] == posteriors))

            likelihoods = [self.log_likelihood(x) for x in chain]
            self.assertTrue(np.all(evals[i, :, 1] == likelihoods))

            priors = [self.log_prior(x) for x in chain]
            self.assertTrue(np.all(evals[i, :, 2] == priors))

        # Test with a loglikelihood
        mcmc = pints.MCMCController(self.log_likelihood, n_chains, xs)
        mcmc.set_max_iterations(n_iterations)
        mcmc.set_log_to_screen(False)
        mcmc.set_log_pdf_storage(True)
        chains = mcmc.run()
        evals = mcmc.log_pdfs()
        self.assertEqual(evals.shape, (n_chains, n_iterations))
        for i, chain in enumerate(chains):
            likelihoods = [self.log_likelihood(x) for x in chain]
            self.assertTrue(np.all(evals[i] == likelihoods))

        # Test with a sensitivity-using method
        mcmc = pints.MCMCController(
            self.log_posterior, n_chains, xs, method=pints.MALAMCMC)
        mcmc.set_max_iterations(n_iterations)
        mcmc.set_log_to_screen(False)
        mcmc.set_log_pdf_storage(True)
        chains = mcmc.run()
        evals = mcmc.log_pdfs()
        self.assertEqual(len(evals.shape), 3)
        self.assertEqual(evals.shape[0], n_chains)
        self.assertEqual(evals.shape[1], n_iterations)
        self.assertEqual(evals.shape[2], 3)
        for i, chain in enumerate(chains):
            posteriors = [self.log_posterior(x) for x in chain]
            self.assertTrue(np.all(evals[i, :, 0] == posteriors))
            likelihoods = [self.log_likelihood(x) for x in chain]
            self.assertTrue(np.all(evals[i, :, 1] == likelihoods))
            priors = [self.log_prior(x) for x in chain]
            self.assertTrue(np.all(evals[i, :, 2] == priors))

        # Test disabling again
        mcmc = pints.MCMCController(self.log_posterior, n_chains, xs)
        mcmc.set_max_iterations(n_iterations)
        mcmc.set_log_to_screen(False)
        mcmc.set_log_pdf_storage(True)
        mcmc.set_log_pdf_storage(False)
        chains = mcmc.run()
        self.assertIsNone(mcmc.log_pdfs())

    def test_log_pdf_storage_in_memory_multi(self):
        # Test storing evaluations in memory, with a multi-chain method

        # Set up test problem
        x0 = np.array(self.real_parameters) * 1.05
        x1 = np.array(self.real_parameters) * 1.15
        x2 = np.array(self.real_parameters) * 0.95
        x3 = np.array(self.real_parameters) * 0.95
        x4 = np.array(self.real_parameters) * 0.95
        xs = [x0, x1, x2, x3, x4]
        n_chains = len(xs)
        n_iterations = 100
        meth = pints.DifferentialEvolutionMCMC

        # Test with multi-chain method
        mcmc = pints.MCMCController(
            self.log_posterior, n_chains, xs, method=meth)
        mcmc.set_max_iterations(n_iterations)
        mcmc.set_log_to_screen(False)
        mcmc.set_log_pdf_storage(True)
        chains = mcmc.run()

        # Test shape of returned array
        evals = mcmc.log_pdfs()
        self.assertEqual(len(evals.shape), 3)
        self.assertEqual(evals.shape[0], n_chains)
        self.assertEqual(evals.shape[1], n_iterations)
        self.assertEqual(evals.shape[2], 3)

        # Test returned values
        for i, chain in enumerate(chains):
            posteriors = [self.log_posterior(x) for x in chain]
            self.assertTrue(np.all(evals[i, :, 0] == posteriors))

            likelihoods = [self.log_likelihood(x) for x in chain]
            self.assertTrue(np.all(evals[i, :, 1] == likelihoods))

            priors = [self.log_prior(x) for x in chain]
            self.assertTrue(np.all(evals[i, :, 2] == priors))

        # Test with a sensitivity-using method
        # We don't have any of these!
        mcmc = pints.MCMCController(
            self.log_posterior, n_chains, xs,
            method=FakeMultiChainSamplerWithSensitivities)
        mcmc.set_max_iterations(5)
        mcmc.set_log_to_screen(False)
        mcmc.set_log_pdf_storage(True)
        chains = mcmc.run()
        evals = mcmc.log_pdfs()
        self.assertEqual(len(evals.shape), 3)
        self.assertEqual(evals.shape[0], n_chains)
        self.assertEqual(evals.shape[1], n_iterations)
        self.assertEqual(evals.shape[2], 3)

        # Test with a loglikelihood
        mcmc = pints.MCMCController(
            self.log_likelihood, n_chains, xs, method=meth)
        mcmc.set_max_iterations(n_iterations)
        mcmc.set_log_to_screen(False)
        mcmc.set_log_pdf_storage(True)
        chains = mcmc.run()
        evals = mcmc.log_pdfs()
        self.assertEqual(evals.shape, (n_chains, n_iterations))
        for i, chain in enumerate(chains):
            likelihoods = [self.log_likelihood(x) for x in chain]
            self.assertTrue(np.all(evals[i] == likelihoods))

    def test_log_pdf_storage_in_memory_single_complex(self):
        # Test storing evaluations in memory, with a single-chain method that
        # does tricky things, e.g. PopulationMCMC maintains internal chains
        # that it swaps around, causing a situation where the last evaluated
        # point on an acceptance step may not be the main chain's point!

        # Set up test problem
        x0 = np.array(self.real_parameters) * 1.05
        x1 = np.array(self.real_parameters) * 1.15
        x2 = np.array(self.real_parameters) * 0.95
        xs = [x0, x1, x2]
        n_chains = len(xs)
        n_iterations = 100

        # Single-chain method, using a logposterior
        mcmc = pints.MCMCController(
            self.log_posterior, n_chains, xs, method=pints.PopulationMCMC)
        mcmc.set_max_iterations(n_iterations)
        mcmc.set_log_to_screen(False)
        mcmc.set_log_pdf_storage(True)
        chains = mcmc.run()

        # Test shape of returned array
        evals = mcmc.log_pdfs()
        self.assertEqual(len(evals.shape), 3)
        self.assertEqual(evals.shape[0], n_chains)
        self.assertEqual(evals.shape[1], n_iterations)
        self.assertEqual(evals.shape[2], 3)

        # Test returned values
        for i, chain in enumerate(chains):
            posteriors = [self.log_posterior(x) for x in chain]
            self.assertTrue(np.all(evals[i, :, 0] == posteriors))

            likelihoods = [self.log_likelihood(x) for x in chain]
            self.assertTrue(np.all(evals[i, :, 1] == likelihoods))

            priors = [self.log_prior(x) for x in chain]
            self.assertTrue(np.all(evals[i, :, 2] == priors))

    def test_deprecated_alias(self):

        with warnings.catch_warnings(record=True) as w:
            mcmc = pints.MCMCSampling(
                self.log_posterior, 1, [self.real_parameters])
        self.assertEqual(len(w), 1)
        self.assertIn('deprecated', str(w[-1].message))
        self.assertIsInstance(mcmc, pints.MCMCController)

    def test_exception_on_multi_use(self):
        # Controller should raise an exception if use multiple times

        # Test simple run
        n_chains = 1
        n_iterations = 10
        x0 = np.array(self.real_parameters) * 1.1
        xs = [x0]
        mcmc = pints.MCMCController(self.log_posterior, n_chains, xs)
        mcmc.set_max_iterations(n_iterations)
        mcmc.set_log_to_screen(False)
        mcmc.run()
        with self.assertRaisesRegex(
                RuntimeError, 'Controller is valid for single use only'):
            mcmc.run()

    def test_post_run_statistics(self):
        # Test method to obtain post-run statistics

        # Set up test problem
        x0 = np.array(self.real_parameters) * 1.05
        x1 = np.array(self.real_parameters) * 1.15
        x2 = np.array(self.real_parameters) * 0.95
        xs = [x0, x1, x2]

        mcmc = pints.MCMCController(self.log_posterior, len(xs), xs)
        mcmc.set_initial_phase_iterations(5)
        mcmc.set_max_iterations(10)
        mcmc.set_log_to_screen(False)
        mcmc.set_log_to_file(False)

        # Before run, methods return None
        self.assertIsNone(mcmc.time())

        t = pints.Timer()
        mcmc.run()
        t_upper = t.time()

        # Check post-run output
        self.assertIsInstance(mcmc.time(), float)
        self.assertGreater(mcmc.time(), 0)
        self.assertGreater(t_upper, mcmc.time())

        # Tets number of evaluations is a realistic number (should be 30 for
        # a simple method)
        self.assertEqual(mcmc.n_evaluations(), 30)


class TestMCMCControllerLogging(unittest.TestCase):
    """
    Test logging to disk and screen.

    Logging is a mechanism for getting feedback on the progress, and possibly
    for analysing progress afterwards. This is distinct from storage of chains
    and evaluations, which isn't tested here.
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
        cls.log_likelihood = pints.GaussianLogLikelihood(problem)

        # Create an un-normalised log-posterior (log-likelihood + log-prior)
        cls.log_posterior = pints.LogPosterior(
            cls.log_likelihood, cls.log_prior)

        # Generate some random starting points
        np.random.seed(1)
        cls.xs = []
        for i in range(3):
            f = 0.9 + 0.2 * np.random.rand()
            cls.xs.append(np.array(cls.real_parameters) * f)
        cls.nchains = len(cls.xs)

    def test_writing_chains_only(self):
        # Test writing chains - but not evals - to disk.

        mcmc = pints.MCMCController(self.log_posterior, self.nchains, self.xs)
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
                chains2 = np.array(io.load_samples(cpath, self.nchains))
                self.assertTrue(np.all(chains1 == chains2))

            text = c.text()
            self.assertIn('Writing chains to', text)
            self.assertIn('chain_0.csv', text)
            self.assertNotIn('Writing evaluations to', text)
            self.assertNotIn('evals_0.csv', text)

        # Test transformation
        logt = pints.LogTransformation(len(self.xs[0]))
        mcmc = pints.MCMCController(self.log_posterior, self.nchains, self.xs,
                                    transformation=logt)
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
                chains2 = np.array(io.load_samples(cpath, self.nchains))
                self.assertTrue(np.all(chains1 == chains2))

                # Test files contain inverse transformed samples
                # log-transformation of the parameter in [0.01, 0.02] will
                # always be negative values, so checking it larger than zero
                # make sure it's transformed back to the model space.
                self.assertTrue(np.all(chains2 > 0))

            text = c.text()
            self.assertIn('Writing chains to', text)
            self.assertIn('chain_0.csv', text)
            self.assertNotIn('Writing evaluations to', text)
            self.assertNotIn('evals_0.csv', text)

    def test_writing_chains_only_no_memory_single(self):
        # Test writing chains - but not evals - to disk, without storing chains
        # in memory, using a single-chain method.

        mcmc = pints.MCMCController(self.log_posterior, self.nchains, self.xs)
        mcmc.set_initial_phase_iterations(5)
        mcmc.set_max_iterations(20)
        mcmc.set_log_to_screen(True)
        mcmc.set_log_to_file(False)
        mcmc.set_chain_storage(False)

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

                # Test chains weren't returned in memory
                self.assertIsNone(chains1)

                # Test disk contains chains
                import pints.io as io
                chains2 = np.array(io.load_samples(cpath, self.nchains))
                self.assertEqual(
                    chains2.shape, (self.nchains, 20, len(self.xs)))

            text = c.text()
            self.assertIn('Writing chains to', text)
            self.assertIn('chain_0.csv', text)
            self.assertNotIn('Writing evaluations to', text)
            self.assertNotIn('evals_0.csv', text)

        # Test transformation
        logt = pints.LogTransformation(len(self.xs[0]))
        mcmc = pints.MCMCController(self.log_posterior, self.nchains, self.xs,
                                    transformation=logt)
        mcmc.set_initial_phase_iterations(5)
        mcmc.set_max_iterations(20)
        mcmc.set_log_to_screen(True)
        mcmc.set_log_to_file(False)
        mcmc.set_chain_storage(False)

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

                # Test chains weren't returned in memory
                self.assertIsNone(chains1)

                # Test disk contains chains
                import pints.io as io
                chains2 = np.array(io.load_samples(cpath, self.nchains))
                self.assertEqual(
                    chains2.shape, (self.nchains, 20, len(self.xs)))

                # Test files contain inverse transformed samples
                # log-transformation of the parameter in [0.01, 0.02] will
                # always be negative values, so checking it larger than zero
                # make sure it's transformed back to the model space.
                self.assertTrue(np.all(chains2 > 0))

            text = c.text()
            self.assertIn('Writing chains to', text)
            self.assertIn('chain_0.csv', text)
            self.assertNotIn('Writing evaluations to', text)
            self.assertNotIn('evals_0.csv', text)

    def test_writing_chains_only_no_memory_multi(self):
        # Test writing chains - but not evals - to disk, without storing chains
        # in memory, using a multi-chain method.

        mcmc = pints.MCMCController(
            self.log_posterior, self.nchains, self.xs,
            method=pints.DifferentialEvolutionMCMC)
        mcmc.set_max_iterations(20)
        mcmc.set_log_to_screen(True)
        mcmc.set_log_to_file(False)
        mcmc.set_chain_storage(False)

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

                # Test chains weren't returned in memory
                self.assertIsNone(chains1)

                # Test disk contains chains
                import pints.io as io
                chains2 = np.array(io.load_samples(cpath, self.nchains))
                self.assertEqual(
                    chains2.shape, (self.nchains, 20, len(self.xs)))

            text = c.text()
            self.assertIn('Writing chains to', text)
            self.assertIn('chain_0.csv', text)

    def test_writing_priors_and_likelihoods(self):
        # Test writing priors and loglikelihoods - not chains - to disk.

        mcmc = pints.MCMCController(self.log_posterior, self.nchains, self.xs)
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
                evals2 = np.array(io.load_samples(epath, self.nchains))
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

    def test_writing_chains_likelihoods_and_priors_single(self):
        # Test writing chains, likelihoods, and priors to disk, using a single
        # chain method.

        mcmc = pints.MCMCController(self.log_posterior, self.nchains, self.xs)
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
                chains2 = np.array(io.load_samples(cpath, self.nchains))
                self.assertTrue(np.all(chains1 == chains2))

                # Test eval files contain the correct values
                evals2 = np.array(io.load_samples(epath, self.nchains))
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

    def test_writing_chains_likelihoods_and_priors_multi(self):
        # Test writing chains, likelihoods, and priors to disk, using a multi
        # chain method.

        mcmc = pints.MCMCController(
            self.log_posterior, self.nchains, self.xs,
            method=pints.DifferentialEvolutionMCMC)
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
                chains2 = np.array(io.load_samples(cpath, self.nchains))
                self.assertTrue(np.all(chains1 == chains2))

                # Test eval files contain the correct values
                evals2 = np.array(io.load_samples(epath, self.nchains))
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

    def test_writing_chains_and_likelihoods_single(self):
        # Test writing chains and likelihoods to disk, using a single chain
        # method.

        mcmc = pints.MCMCController(self.log_likelihood, self.nchains, self.xs)
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
                chains2 = np.array(io.load_samples(cpath, self.nchains))
                self.assertTrue(np.all(chains1 == chains2))

                # Test eval files contain the correct values
                evals2 = np.array(io.load_samples(epath, self.nchains))
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

    def test_writing_chains_likelihoods_and_priors_one_chain(self):
        # Test with a single chain.

        nchains = 1
        mcmc = pints.MCMCController(self.log_posterior, nchains, self.xs[:1])
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

    def test_disabling_disk_storage(self):
        # Test if storage can be enabled and then disabled again.
        mcmc = pints.MCMCController(self.log_posterior, self.nchains, self.xs)
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


class SumDistribution(pints.LogPDF):
    """
    Distribution where p(x) = 1 + sum(x)``, used in testing writing samples and
    evaluations to disk.
    """

    def __init__(self, n_parameters):
        self._n_parameters = n_parameters

    def __call__(self, x):
        return 1 + np.sum(x)

    def n_parameters(self):
        return self._n_parameters


class SingleListSampler(pints.SingleChainMCMC):
    """
    Returns predetermined samples from a list, used in testing writing samples
    and evaluations to disk.

    First sample can't be None.
    """

    def set_chain(self, chain):
        self._chain = list(chain)
        self._i = 0
        self._n = len(self._chain)

    def ask(self):
        x = self._chain[self._i] if self._i < self._n else None
        if x is None:
            x = self._chain[0]
        return x

    def name(self):
        return 'SingleListSampler'

    def tell(self, fx):
        x = self._chain[self._i] if self._i < self._n else None
        self._i += 1
        return None if x is None else (x, fx, True)


class MultiListSampler(pints.MultiChainMCMC):
    """
    Returns predetermined samples from a list of chains, used in testing
    writing samples and evaluations to disk.

    Adding a ``None`` in the first list at any point will cause ``None`` to be
    returned (for all chains) at that iteration.
    First sample can't be None.
    """

    def set_chains(self, chains):
        self._chains = [list(x) for x in chains]
        self._i = 0
        self._n = len(self._chains[0])

    def ask(self):
        x = [chain[self._i] for chain in self._chains]
        if x[0] is None:
            x = [chain[0] for chain in self._chains]
        return x

    def name(self):
        return 'MultiListSampler'

    def tell(self, fx):
        x = None
        if self._i < self._n:
            x = [chain[self._i] for chain in self._chains]
            if x[0] is None:
                x = None
            self._i += 1
        return None if x is None else (x, fx, np.array([True] * self._n))


class FakeMultiChainSamplerWithSensitivities(pints.MultiChainMCMC):
    """
    Fake sampler that pretends to be a multi-chain method that uses
    sensitivities. At the moment (2024-12-18) we don't have these in PINTS, but
    we need to check that code potentially handling this type of sampler exists
    or raises not-implemented errors.
    """
    def ask(self):
        self._xs = self._x0 + 1e-3 * np.random.uniform(
            size=(self._n_chains, self._n_parameters))
        return self._xs

    def current_log_pdfs(self):
        return self._fxs

    def tell(self, fxs):
        self._fxs = fxs
        return self._xs, self._fxs, [True] * self._n_chains


class TestMCMCControllerSingleChainStorage(unittest.TestCase):
    """
    Tests storage of samples and evaluations to disk, running with a
    single-chain MCMC method.
    """

    def go(self, chains):
        """
        Run with a given list of expected chains, return obtained output.
        """

        # Filter nones to get expected output
        expected = np.array(
            [[x for x in chain if x is not None] for chain in chains])

        # Get initial position
        x0 = [chain[0] for chain in chains]

        # Create log pdf
        f = SumDistribution(len(x0[0]))

        # Get expected evaluations
        exp_evals = np.array(
            [[[f(x)] for x in chain] for chain in expected])

        # Set up controller
        nc = len(x0)
        mcmc = pints.MCMCController(f, nc, x0, method=SingleListSampler)
        mcmc.set_log_to_screen(False)
        mcmc.set_max_iterations(len(expected[0]))

        # Pass chains to samplers
        for i, sampler in enumerate(mcmc.samplers()):
            sampler.set_chain(chains[i])

        # Run, while logging to disk
        with TemporaryDirectory() as d:
            # Store chains
            chain_path = d.path('chain.csv')
            mcmc.set_chain_filename(chain_path)

            # Store log pdfs
            evals_path = d.path('evals.csv')
            mcmc.set_log_pdf_filename(evals_path)

            # Run
            obtained = mcmc.run()

            # Load chains and log_pdfs
            disk_samples = np.array(pints.io.load_samples(chain_path, nc))
            disk_evals = np.array(pints.io.load_samples(evals_path, nc))

        # Return expected and obtained values
        return expected, obtained, disk_samples, exp_evals, disk_evals

    def test_one_sampler_no_nones(self):
        # One single-chain sampler, no None objects.
        chains1 = [[[2], [4], [6], [3], [5]]]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = self.go(chains1)
        self.assertEqual(chains2.shape, (1, 5, 1))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (1, 5, 1))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (1, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

        chains1 = [[[1, 2], [4, 3], [6, 1], [2, 2], [5, 7]]]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = self.go(chains1)
        self.assertEqual(chains2.shape, (1, 5, 2))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (1, 5, 2))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (1, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

    def test_one_sampler_with_nones(self):
        # One single-chain sampler, with None objects.
        chains1 = [[[1], [3], None, [1], None, [5], [2]]]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = self.go(chains1)
        self.assertEqual(chains2.shape, (1, 5, 1))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (1, 5, 1))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (1, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

        chains1 = [[[1], [3], None, [1], None, [5], None, None, None, [2]]]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = self.go(chains1)
        self.assertEqual(chains2.shape, (1, 5, 1))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (1, 5, 1))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (1, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

        chains1 = [[[1, 2], [4, 3], None, [6, 1], None, [2, 2], [5, 7]]]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = self.go(chains1)
        self.assertEqual(chains2.shape, (1, 5, 2))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (1, 5, 2))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (1, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

    def test_multiple_samplers_no_nones(self):
        # Multiple single-chain samplers, no None objects.
        chains1 = [
            [[2], [4], [6], [3], [5]],
            [[5], [1], [3], [3], [2]],
        ]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = self.go(chains1)
        self.assertEqual(chains2.shape, (2, 5, 1))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (2, 5, 1))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (2, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

        chains1 = [
            [[1, 2], [4, 3], [6, 1], [2, 2], [5, 7]],
            [[4, 3], [1, 1], [3, 5], [1, 4], [4, 7]],
        ]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = self.go(chains1)
        self.assertEqual(chains2.shape, (2, 5, 2))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (2, 5, 2))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (2, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

    def test_multiple_samplers_same_index_nones(self):
        # Multiple single-chain samplers, None at same index.
        chains1 = [
            [[2], None, None, [4], [6], None, [3], None, None, [5]],
            [[5], None, None, [1], [3], None, [3], None, None, [2]],
        ]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = self.go(chains1)
        self.assertEqual(chains2.shape, (2, 5, 1))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (2, 5, 1))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (2, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

        chains1 = [
            [[1, 2], None, [4, 3], [6, 1], None, None, None, [2, 2], [5, 7]],
            [[4, 3], None, [1, 1], [3, 5], None, None, None, [1, 4], [4, 7]],
        ]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = self.go(chains1)
        self.assertEqual(chains2.shape, (2, 5, 2))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (2, 5, 2))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (2, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

    def test_multiple_samplers_mixed_index_nones(self):
        # Multiple single-chain samplers, None at different indices.
        chains1 = [
            [[2], None, [4], [6], None, [3], [5], None, None, None],
            [[5], None, None, None, [1], [3], None, [3], None, [2]],
        ]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = self.go(chains1)
        self.assertEqual(chains2.shape, (2, 5, 1))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (2, 5, 1))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (2, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

        chains1 = [
            [[1, 2], [4, 3], [6, 1], None, [5, 7], [2, 2], None, None],
            [[4, 3], None, None, [1, 1], [3, 5], None, [1, 4], [4, 7]],
        ]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = self.go(chains1)
        self.assertEqual(chains2.shape, (2, 5, 2))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (2, 5, 2))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (2, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))


class TestMCMCControllerMultiChainStorage(unittest.TestCase):
    """
    Tests storage of samples and evaluations to disk, running with a
    multi-chain MCMC method.
    """

    def go(self, chains):
        """
        Run with a given list of expected chains, return obtained output.
        """

        # Filter nones to get expected output
        expected = np.array(
            [[x for x in chain if x is not None] for chain in chains])

        # Get initial positions
        x0 = [chain[0] for chain in chains]

        # Create log pdf
        f = SumDistribution(len(x0[0]))

        # Get expected evaluations
        exp_evals = np.array(
            [[[f(x)] for x in chain] for chain in expected])

        # Set up controller
        nc = len(x0)
        mcmc = pints.MCMCController(f, nc, x0, method=MultiListSampler)
        mcmc.set_log_to_screen(False)
        mcmc.set_max_iterations(len(expected[0]))

        # Pass chains to sampler
        mcmc.sampler().set_chains(chains)

        # Run, while logging to disk
        with TemporaryDirectory() as d:
            # Store chains
            chain_path = d.path('chain.csv')
            mcmc.set_chain_filename(chain_path)

            # Store log pdfs
            evals_path = d.path('evals.csv')
            mcmc.set_log_pdf_filename(evals_path)

            # Run
            obtained = mcmc.run()

            # Load chains and log_pdfs
            disk_samples = np.array(pints.io.load_samples(chain_path, nc))
            disk_evals = np.array(pints.io.load_samples(evals_path, nc))

        # Return expected and obtained values
        return expected, obtained, disk_samples, exp_evals, disk_evals

    def test_single_chain_no_nones(self):
        # Test with a single chain, no None objects.
        chains1 = [[[2], [2], [6], [3], [0.5]]]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = self.go(chains1)
        self.assertEqual(chains2.shape, (1, 5, 1))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (1, 5, 1))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (1, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

        chains1 = [[[1, 2], [2, 4], [3, 6], [8, 8], [1, 2]]]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = self.go(chains1)
        self.assertEqual(chains2.shape, (1, 5, 2))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (1, 5, 2))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (1, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

    def test_multi_chain_no_nones(self):
        # Test with three chains, no None objects.
        chains1 = [
            [[1, 2], [2, 4], [3, 6], [8, 8], [1, 2]],
            [[2, 3], [3, 5], [4, 7], [9, 8], [3, 2]],
            [[3, 4], [4, 6], [5, 8], [8, 3], [2, 7]],
        ]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = self.go(chains1)
        self.assertEqual(chains2.shape, (3, 5, 2))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (3, 5, 2))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (3, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

    def test_single_chain_with_nones(self):
        # Test with a single chain, some None objects.
        chains1 = [[[2], None, [2], None, [6], None, None, [3], [0.5]]]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = self.go(chains1)
        self.assertEqual(chains2.shape, (1, 5, 1))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (1, 5, 1))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (1, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

    def test_multi_chain_with_nones(self):
        # Test with two chains, some None objects
        chains1 = [
            [[1, 2], [2, 4], None, [3, 6], None, None, [8, 8], None, [1, 2]],
            [[3, 4], [4, 6], None, [5, 8], None, None, [8, 3], None, [2, 7]],
        ]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = self.go(chains1)
        self.assertEqual(chains2.shape, (2, 5, 2))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (2, 5, 2))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (2, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
