#!/usr/bin/env python3
#
# Tests the nested sampling controller.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import re
import unittest
import numpy as np

import pints
import pints.toy

from shared import StreamCapture, TemporaryDirectory


class TestNestedController(unittest.TestCase):
    """
    Unit (not functional!) tests for :class:`NestedController`.
    """

    @classmethod
    def setUpClass(cls):
        """ Prepare for the test. """
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
            [0.01, 400],
            [0.02, 600]
        )

        # Create a log-likelihood
        cls.log_likelihood = pints.GaussianKnownSigmaLogLikelihood(
            problem, cls.noise)

    def test_quick_run(self):
        # Test a single run.

        sampler = pints.NestedController(
            self.log_likelihood, self.log_prior)
        sampler.set_n_posterior_samples(10)
        sampler.set_iterations(50)
        sampler.set_log_to_screen(False)

        # Time before run is None
        self.assertIsNone(sampler.time())

        t = pints.Timer()
        samples = sampler.run()
        t_upper = t.time()

        # Check output: Note n returned samples = n posterior samples
        self.assertEqual(samples.shape, (10, 2))

        # Time after run is greater than zero
        self.assertIsInstance(sampler.time(), float)
        self.assertGreater(sampler.time(), 0)
        self.assertGreater(t_upper, sampler.time())

    def test_construction_errors(self):
        # Tests if invalid constructor calls are picked up.

        # First arg must be a log likelihood
        self.assertRaisesRegex(
            ValueError, 'must extend pints.LogLikelihood',
            pints.NestedController, 'hello', self.log_prior)

        # First arg must be a log prior
        self.assertRaisesRegex(
            ValueError, 'must extend pints.LogPrior',
            pints.NestedController,
            self.log_likelihood, self.log_likelihood)

        # Both must have same number of parameters
        log_prior = pints.UniformLogPrior([0.01, 400, 1], [0.02, 600, 3])
        self.assertRaisesRegex(
            ValueError, 'same number of parameters',
            pints.NestedController, self.log_likelihood, log_prior)

        # test that ellipsoidal sampling used by default
        sampler = pints.NestedController(self.log_likelihood, self.log_prior)
        self.assertEqual(sampler._sampler.name(), 'Nested ellipsoidal sampler')
        self.assertRaisesRegex(
            ValueError,
            'Given method must extend pints.NestedSampler.',
            pints.NestedController,
            self.log_likelihood, self.log_prior,
            pints.DifferentialEvolutionMCMC)

        self.assertRaisesRegex(
            ValueError,
            'Given method must extend pints.NestedSampler.',
            pints.NestedController,
            self.log_likelihood, self.log_prior,
            0.0)

    def test_parallel(self):
        # Test running sampling with parallisation.

        sampler = pints.NestedController(self.log_likelihood,
                                         self.log_prior)
        # Test with auto-detected number of worker processes
        self.assertFalse(sampler.parallel())
        sampler.set_parallel(True)
        self.assertTrue(sampler.parallel())
        sampler.set_iterations(10)
        sampler.set_log_to_screen(False)
        sampler.run()

        # Test with fixed number of worker processes
        sampler = pints.NestedController(
            self.log_likelihood, self.log_prior)
        sampler.set_parallel(4)
        sampler.set_log_to_screen(False)
        self.assertEqual(sampler.parallel(), 4)
        sampler.run()

    def test_logging(self):
        # Tests logging to screen and file.

        # No logging
        with StreamCapture() as c:
            sampler = pints.NestedController(
                self.log_likelihood, self.log_prior)
            sampler.set_n_posterior_samples(2)
            sampler.set_iterations(10)
            sampler.set_log_to_screen(False)
            sampler.set_log_to_file(False)
            samples, margin = sampler.run()
        self.assertEqual(c.text(), '')

        # Log to screen
        with StreamCapture() as c:
            sampler = pints.NestedController(
                self.log_likelihood, self.log_prior)
            sampler.set_n_posterior_samples(2)
            sampler.set_iterations(20)
            sampler.set_log_to_screen(True)
            sampler.set_log_to_file(False)
            samples, margin = sampler.run()
        lines = c.text().splitlines()
        self.assertEqual(lines[0], 'Running Nested ellipsoidal sampler')
        self.assertEqual(lines[1], 'Number of active points: 400')
        self.assertEqual(lines[2], 'Total number of iterations: 20')
        self.assertEqual(lines[3], 'Total number of posterior samples: 2')
        self.assertEqual(lines[4], ('Iter. Eval. Time m:s Delta_log(z) ' +
                                    'Acceptance rate'))
        pattern = re.compile('[0-9]+[ ]+[0-9]+[ ]+[0-9]{1}:[0-9]{2}.[0-9]{1}')
        for line in lines[5:]:
            self.assertTrue(pattern.match(line))
        self.assertEqual(len(lines), 28)

        # Log to file
        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                filename = d.path('test.txt')
                sampler = pints.NestedController(
                    self.log_likelihood, self.log_prior)
                sampler.set_n_posterior_samples(2)
                sampler.set_iterations(10)
                sampler.set_log_to_screen(False)
                sampler.set_log_to_file(filename)
                samples, margin = sampler.run()
                with open(filename, 'r') as f:
                    lines = f.read().splitlines()
            self.assertEqual(c.text(), '')
        self.assertEqual(len(lines), 23)
        self.assertEqual(lines[0], ('Iter. Eval. Time m:s Delta_log(z) ' +
                                    'Acceptance rate'))
        pattern = re.compile('[0-9]+[ ]+[0-9]+[ ]+[0-9]{1}:[0-9]{2}.[0-9]{1}')
        for line in lines[5:]:
            self.assertTrue(pattern.match(line))

    def test_settings_check(self):
        # Tests the settings check at the start of a run.
        sampler = pints.NestedController(
            self.log_likelihood, self.log_prior)
        sampler.set_n_posterior_samples(2)
        sampler.set_iterations(10)
        sampler.set_log_to_screen(False)
        sampler.run()

    def test_nested_sampler(self):
        # Tests `NestedSampler`.
        sampler = pints.NestedSampler(self.log_prior)
        self.assertTrue(not sampler.needs_initial_phase())

    def test_getters_and_setters(self):
        # Tests various get() and set() methods.
        sampler = pints.NestedController(
            self.log_likelihood, self.log_prior)

        # Iterations
        x = sampler.iterations() + 1
        self.assertNotEqual(sampler.iterations(), x)
        sampler.set_iterations(x)
        self.assertEqual(sampler.iterations(), x)
        self.assertRaisesRegex(
            ValueError, 'negative', sampler.set_iterations, -1)

        # Posterior samples
        x = sampler.n_posterior_samples() + 1
        self.assertNotEqual(sampler.n_posterior_samples(), x)
        sampler.set_n_posterior_samples(x)
        self.assertEqual(sampler.n_posterior_samples(), x)
        self.assertRaisesRegex(
            ValueError, 'greater than zero',
            sampler.set_n_posterior_samples, 0)
        self.assertRaises(ValueError, sampler.sample_from_posterior, 0)

        # Marginal likelihood threshold
        self.assertRaises(ValueError,
                          sampler.set_marginal_log_likelihood_threshold,
                          0)
        sampler.set_marginal_log_likelihood_threshold(3.0)
        self.assertEqual(sampler.marginal_log_likelihood_threshold(), 3.0)

        # Acive points
        sampler.set_iterations(100)
        sampler.set_log_to_screen(False)
        sampler.run()
        active_points = sampler.active_points()
        self.assertEqual(active_points.shape[0], 400)

        # effective sample size and log-likelihood vector
        ess1 = sampler.effective_sample_size()
        logLikelihood1 = sampler.log_likelihood_vector()
        self.assertEqual(len(logLikelihood1), 400 + 100)
        self.assertTrue(ess1 > 0)
        sampler = pints.NestedController(
            self.log_likelihood, self.log_prior)
        iter = 2000
        sampler.set_iterations(iter)
        sampler.set_n_posterior_samples(100)
        sampler.set_log_to_screen(False)
        sampler.run()
        ess2 = sampler.effective_sample_size()
        self.assertTrue(ess2 > ess1)
        logLikelihood2 = sampler.log_likelihood_vector()
        self.assertEqual(len(logLikelihood2), 400 + iter)

        # marginal likelihood
        ess_sd1 = sampler.marginal_log_likelihood_standard_deviation()
        self.assertTrue(ess_sd1 > 0)
        sampler._log_Z_called = False
        ess_sd2 = sampler.marginal_log_likelihood_standard_deviation()
        self.assertEqual(ess_sd1, ess_sd2)

        # number of posterior samples
        m_posterior_samples = sampler.posterior_samples()
        self.assertEqual(m_posterior_samples.shape[0], 100)

        # prior space
        prior_space = sampler.prior_space()
        self.assertEqual(len(prior_space), iter + 1)
        for elem in prior_space:
            self.assertTrue(elem >= 0)
            self.assertTrue(elem <= 1)

        # Acive points
        sampler = pints.NestedController(
            self.log_likelihood, self.log_prior)
        sampler.set_iterations(100)
        sampler.set_log_to_screen(False)
        sampler.set_parallel(2)
        sampler.run()
        active_points = sampler.active_points()
        self.assertEqual(active_points.shape[0], 400)
        inactive_points = sampler.inactive_points()
        self.assertEqual(inactive_points.shape[0], 100)

    def test_nones(self):
        # test handing of nones
        # test that None is returned
        sampler = pints.NestedEllipsoidSampler(self.log_prior)
        pts = sampler.ask(1)
        fx = np.nan
        sample, other = sampler.tell(fx)
        self.assertEqual(sample, None)

        # test that None is returned
        sampler = pints.NestedEllipsoidSampler(self.log_prior)
        pts = sampler.ask(1)
        fx = [np.nan, np.nan]
        sample, other = sampler.tell(fx)
        self.assertEqual(sample, None)

        # test if fx has one None and one non-none
        pts = sampler.ask(2)
        fx = [np.nan, -20]
        sample, other = sampler.tell(fx)
        self.assertEqual(sample[0], pts[1][0])

    def test_early_termination(self):
        # tests that nested sampling terminates early with a large
        # threshold
        sampler = pints.NestedController(self.log_likelihood,
                                         self.log_prior)
        # Test with auto-detected number of worker processes
        self.assertFalse(sampler.parallel())
        sampler.set_parallel(True)
        self.assertTrue(sampler.parallel())
        sampler.set_iterations(200)
        sampler.set_log_to_screen(False)
        sampler.set_marginal_log_likelihood_threshold(100000)
        sampler.run()
        m_inactive = sampler.inactive_points()
        self.assertTrue(m_inactive.shape[0] < 200)

    def test_exception_on_multi_use(self):
        # Controller should raise an exception if use multiple times

        sampler = pints.NestedController(
            self.log_likelihood, self.log_prior)
        sampler.set_n_posterior_samples(2)
        sampler.set_iterations(10)
        sampler.set_log_to_screen(False)
        sampler.run()
        with self.assertRaisesRegex(
                RuntimeError, 'Controller is valid for single use only'):
            sampler.run()


if __name__ == '__main__':
    unittest.main()
