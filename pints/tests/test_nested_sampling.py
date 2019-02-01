#!/usr/bin/env python
#
# Tests the basic methods of the nested sampling routines.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import re
import unittest
import numpy as np

import pints
import pints.toy

from shared import StreamCapture, TemporaryDirectory

# Unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp

debug = False


class TestNestedRejectionSampler(unittest.TestCase):
    """
    Unit (not functional!) tests for :class:`NestedRejectionSampler`.
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

    def test_setup_and_parameters_nested_sampler(self):
        """ Test setup of nested sampler"""

        sampler = pints.NestedRejectionSampler(self.log_prior)

        # Test initial constructors
        self.assertEqual(sampler._running_log_likelihood, -float('Inf'))
        self.assertEqual(sampler._proposed, None)
        self.assertEqual(sampler._n_active_points, 400)
        self.assertEqual(sampler._dimension, 2)
        self.assertSequenceEqual(sampler._m_active.shape,
                                 (sampler._n_active_points,
                                  sampler._dimension + 1))
        self.assertEqual(sampler._min_index, None)

        # Test functions
        self.assertTrue(not sampler.needs_sensitivities())
        self.assertEqual(sampler.name(), 'Nested Rejection sampler')

        # Generate initial points by sampling from prior
        n_active_points = sampler.n_active_points()
        m_initial = sampler._log_prior.sample(n_active_points)
        v_fx = np.zeros(n_active_points)
        for i in range(n_active_points):
            v_fx[i] = self.log_likelihood(m_initial)
        sampler.initialise_active_points(m_initial, v_fx)
        self.assertTrue(sampler.min_index(), not None)
        m_active = sampler.active_point()
        self.assertSequenceEqual(m_active.shape,
                                 (n_active_points, self._dimension + 1))
        self.assertEqual(np.sum(m_active[:, self._dimension]), np.sum(v_fx))
        self.assertEqual(np.sum(m_active[:, 0]), np.sum(m_initial[:, 0]))
        self.assertEqual(sampler.running_log_likelihood(), v_fx)

        proposed = sampler.ask()
        self.assertTrue(len(proposed), 2)
        self.assertEqual(proposed, sampler._proposed)
        fx = self.log_likelihood(proposed)
        if fx < sampler.running_log_likelihood():
            self.assertEqual(sampler.tell(fx), None)
        else:
            self.assertEqual(sampler.tell(fx), self._proposed)

        self.assertEqual(sampler.tell(float('nan')), None)
        self.assertEqual(sampler.tell(-float('Inf')), None)

        # force a value that tell will accept
        self._proposed = [0.015, 500]
        init_log_likelihood = sampler.running_log_likelihood()
        a_min_index = sampler.min_index()
        fx = self.log_likelihood(sampler._proposed)
        proposed = sampler.tell(fx)
        m_active = sampler.active_points()
        self.assertEqual(sampler._proposed, proposed)
        self.assertSequenceEqual(m_active[:, a_min_index],
                                 np.concatenate((proposed, np.array([fx]))))
        self.assertTrue(sampler.running_log_likelihood() > init_log_likelihood)

        self.assertTrue(not sampler.needs_initial_phase())

    def test_setup_and_parameters_nested_sampling(self):
        """ Test setup of nested sampling """
        sampler = pints.NestedSampling(self.log_likelihood, self.log_prior,
                                       method=pints.NestedRejectionSampler)
        self.assertTrue(not sampler.parallel())

        sampler.set_n_posterior_samples(10)
        sampler.set_iterations(50)
        sampler._sampler.set_n_active_points(50)
        sampler.set_log_to_screen(False)
        samples = sampler.run()
        # Check output: Note n returned samples = n posterior samples
        self.assertEqual(samples.shape, (10, 2))

    def test_construction_errors(self):
        """ Tests if invalid constructor calls are picked up. """

        # First arg must be a log prior
        self.assertRaisesRegex(
            ValueError, 'must extend pints.LogPrior',
            pints.NestedRejectionSampler, self.log_likelihood)

        # Both must have same number of parameters
        log_prior = pints.UniformLogPrior([0.01, 400, 1], [0.02, 600, 3])
        self.assertRaisesRegex(
            ValueError, 'same number of parameters',
            pints.NestedSampling, self.log_likelihood, log_prior)

    def test_logging(self):
        """ Tests logging to screen and file. """

        # No logging
        with StreamCapture() as c:
            sampler = pints.NestedSampling(
                self.log_likelihood, self.log_prior,
                method=pints.NestedRejectionSampler)
            sampler.set_n_posterior_samples(2)
            sampler.set_iterations(10)
            sampler._sampler.set_n_active_points(10)
            sampler.set_log_to_screen(False)
            sampler.set_log_to_file(False)
            samples = sampler.run()
        self.assertEqual(c.text(), '')

        # Log to screen
        with StreamCapture() as c:
            sampler = pints.NestedSampling(
                self.log_likelihood, self.log_prior,
                method=pints.NestedRejectionSampler)
            sampler.set_n_posterior_samples(2)
            sampler.set_iterations(20)
            sampler._sampler.set_n_active_points(10)
            sampler.set_log_to_screen(True)
            sampler.set_log_to_file(False)
            samples = sampler.run()
        lines = c.text().splitlines()
        self.assertEqual(lines[0], 'Running nested rejection sampling')
        self.assertEqual(lines[1], 'Number of active points: 10')
        self.assertEqual(lines[2], 'Total number of iterations: 20')
        self.assertEqual(lines[3], 'Total number of posterior samples: 2')
        self.assertEqual(lines[4], 'Iter. Eval. Time m:s')
        pattern = re.compile('[0-9]+[ ]+[0-9]+[ ]+[0-9]{1}:[0-9]{2}.[0-9]{1}')
        for line in lines[5:]:
            self.assertTrue(pattern.match(line))
        self.assertEqual(len(lines), 11)

        # Log to file
        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                filename = d.path('test.txt')
                sampler = pints.NestedSampling(
                    self.log_likelihood, self.log_prior,
                    method=pints.NestedRejectionSampler)
                sampler.set_n_posterior_samples(2)
                sampler.set_iterations(10)
                sampler._sampler.set_n_active_points(10)
                sampler.set_log_to_screen(False)
                sampler.set_log_to_file(filename)
                samples = sampler.run()
                with open(filename, 'r') as f:
                    lines = f.read().splitlines()
            self.assertEqual(c.text(), '')
        self.assertEqual(len(lines), 6)
        self.assertEqual(lines[0], 'Iter. Eval. Time m:s')
        pattern = re.compile('[0-9]+[ ]+[0-9]+[ ]+[0-9]{1}:[0-9]{2}.[0-9]{1}')
        for line in lines[5:]:
            self.assertTrue(pattern.match(line))

    def test_settings_check(self):
        """
        Tests the settings check at the start of a run.
        """
        sampler = pints.NestedSampling(
            self.log_likelihood, self.log_prior,
            method=pints.NestedRejectionSampler)
        sampler.set_n_posterior_samples(2)
        sampler.set_iterations(10)
        sampler._sampler.set_n_active_points(10)
        sampler.set_log_to_screen(False)
        sampler.run()

        sampler.set_n_posterior_samples(10)
        self.assertRaisesRegex(ValueError, 'exceed 0.25', sampler.run)

    def test_hyper_params(self):
        """
        Tests the hyper parameter interface is working.
        """
        sampler = pints.NestedRejectionSampler(self.log_prior)
        self.assertEqual(sampler.n_hyper_parameters(), 1)
        sampler.set_hyper_parameters([6])
        self.assertEqual(sampler.n_active_points(), 6)

    def test_getters_and_setters(self):
        """
        Tests various get() and set() methods.
        """
        sampler = pints.NestedSampling(
            self.log_likelihood, self.log_prior,
            method=pints.NestedRejectionSampler)

        # Iterations
        x = sampler.iterations() + 1
        self.assertNotEqual(sampler.iterations(), x)
        sampler.set_iterations(x)
        self.assertEqual(sampler.iterations(), x)
        self.assertRaisesRegex(
            ValueError, 'negative', sampler.set_iterations, -1)

        # Active points rate
        x = sampler._sampler.n_active_points() + 1
        self.assertNotEqual(sampler._sampler.n_active_points(), x)
        sampler._sampler.set_n_active_points(x)
        self.assertEqual(sampler._sampler.n_active_points(), x)
        self.assertRaisesRegex(
            ValueError, 'greater than 5',
            sampler._sampler.set_n_active_points, 5)

        # Posterior samples
        x = sampler.n_posterior_samples() + 1
        self.assertNotEqual(sampler.n_posterior_samples(), x)
        sampler.set_n_posterior_samples(x)
        self.assertEqual(sampler.n_posterior_samples(), x)
        self.assertRaisesRegex(
            ValueError, 'greater than zero',
            sampler.set_n_posterior_samples, 0)


class TestNestedEllipsoidSampler(unittest.TestCase):
    """
    Unit (not functional!) tests for :class:`NestedEllipsoidSampler`.
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

    def test_construction_errors(self):
        """ Tests if invalid constructor calls are picked up. """

        # First arg must be a log prior
        self.assertRaisesRegex(
            ValueError, 'must extend pints.LogPrior',
            pints.NestedEllipsoidSampler, 'hiya')

        # First arg must be a log prior
        self.assertRaisesRegex(
            ValueError, 'must extend pints.LogPrior',
            pints.NestedEllipsoidSampler,
            self.log_likelihood)

    def test_quick(self):
        """ Test a single run. """

        sampler = pints.NestedSampling(
            self.log_likelihood, self.log_prior,
            method=pints.NestedEllipsoidSampler)
        sampler.set_n_posterior_samples(10)
        sampler._sampler.set_rejection_samples(20)
        sampler.set_iterations(50)
        sampler._sampler.set_n_active_points(50)
        sampler.set_log_to_screen(False)
        samples = sampler.run()
        # Check output: Note n returned samples = n posterior samples
        self.assertEqual(samples.shape, (10, 2))

    def test_settings_check(self):
        """
        Tests the settings check at the start of a run.
        """
        sampler = pints.NestedSampling(
            self.log_likelihood, self.log_prior,
            method=pints.NestedEllipsoidSampler)
        sampler.set_n_posterior_samples(2)
        sampler._sampler.set_rejection_samples(5)
        sampler.set_iterations(10)
        sampler._sampler.set_n_active_points(10)
        sampler.set_log_to_screen(False)
        sampler.run()

        sampler.set_n_posterior_samples(10)
        self.assertRaisesRegex(ValueError, 'exceed 0.25', sampler.run)
        sampler.set_n_posterior_samples(2)
        sampler.set_iterations(4)

    def test_hyper_params(self):
        """
        Tests the hyper parameter interface is working.
        """
        sampler = pints.NestedEllipsoidSampler(self.log_prior)
        self.assertEqual(sampler.n_hyper_parameters(), 4)
        sampler.set_hyper_parameters([550, 1000, 1.2, 20])
        self.assertEqual(sampler.n_active_points(), 550)
        self.assertEqual(sampler.rejection_samples(), 1000)
        self.assertEqual(sampler.ellipsoid_update_gap(), 20)
        self.assertEqual(sampler.enlargement_factor(), 1.2)

    def test_logging(self):
        """ Tests logging to screen and file. """

        # No logging
        with StreamCapture() as c:
            sampler = pints.NestedSampling(
                self.log_likelihood, self.log_prior,
                method=pints.NestedEllipsoidSampler)
            sampler.set_n_posterior_samples(2)
            sampler._sampler.set_rejection_samples(5)
            sampler.set_iterations(10)
            sampler._sampler.set_n_active_points(10)
            sampler.set_log_to_screen(False)
            sampler.set_log_to_file(False)
            samples = sampler.run()
        self.assertEqual(c.text(), '')

        # Log to screen
        with StreamCapture() as c:
            sampler = pints.NestedSampling(
                self.log_likelihood, self.log_prior,
                method=pints.NestedEllipsoidSampler)
            sampler.set_n_posterior_samples(2)
            sampler._sampler.set_rejection_samples(5)
            sampler.set_iterations(20)
            sampler._sampler.set_n_active_points(10)
            sampler.set_log_to_screen(True)
            sampler.set_log_to_file(False)
            samples = sampler.run()
        lines = c.text().splitlines()
        self.assertEqual(lines[0], 'Running Nested Ellipsoidal Rejection ' +
                                   'sampler')
        self.assertEqual(lines[1], 'Number of active points: 10')
        self.assertEqual(lines[2], 'Total number of iterations: 20')
        self.assertEqual(lines[3], 'Total number of posterior samples: 2')
        self.assertEqual(lines[4], 'Iter. Eval. Time m:s')
        pattern = re.compile('[0-9]+[ ]+[0-9]+[ ]+[0-9]{1}:[0-9]{2}.[0-9]{1}')
        for line in lines[6:]:
            self.assertTrue(pattern.match(line))
        self.assertEqual(len(lines), 12)

        # Log to file
        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                filename = d.path('test.txt')
                pints.NestedSampling(
                    self.log_likelihood, self.log_prior,
                    method=pints.NestedEllipsoidSampler)
                sampler.set_n_posterior_samples(2)
                sampler._sampler.set_rejection_samples(5)
                sampler.set_iterations(10)
                sampler._sampler.set_n_active_points(10)
                sampler.set_log_to_screen(False)
                sampler.set_log_to_file(filename)
                samples = sampler.run()
                with open(filename, 'r') as f:
                    lines = f.read().splitlines()
            self.assertEqual(c.text(), '')
        self.assertEqual(len(lines), 6)
        self.assertEqual(lines[0], 'Iter. Eval. Time m:s')
        pattern = re.compile('[0-9]+[ ]+[0-9]+[ ]+[0-9]{1}:[0-9]{2}.[0-9]{1}')
        for line in lines[5:]:
            self.assertTrue(pattern.match(line))

    def test_getters_and_setters(self):
        """
        Tests various get() and set() methods.
        """
        sampler = pints.NestedSampling(
            self.log_likelihood, self.log_prior,
            method=pints.NestedEllipsoidSampler)

        # Iterations
        x = sampler.iterations() + 1
        self.assertNotEqual(sampler.iterations(), x)
        sampler.set_iterations(x)
        self.assertEqual(sampler.iterations(), x)
        self.assertRaisesRegex(
            ValueError, 'negative', sampler.set_iterations, -1)

        # Active points rate
        x = sampler._sampler.n_active_points() + 1
        sampler._sampler.set_n_active_points(x)
        self.assertEqual(sampler._sampler.n_active_points(), x)
        self.assertRaisesRegex(
            ValueError, 'greater than 5',
            sampler._sampler.set_n_active_points, 5)

        # Posterior samples
        x = sampler.n_posterior_samples() + 1
        sampler.set_n_posterior_samples(x)
        self.assertEqual(sampler.n_posterior_samples(), x)
        self.assertRaisesRegex(
            ValueError, 'greater than zero',
            sampler.set_n_posterior_samples, 0)

        # Enlargement factor
        x = sampler._sampler.enlargement_factor() * 2
        self.assertNotEqual(sampler._sampler.enlargement_factor(), x)
        sampler._sampler.set_enlargement_factor(x)
        self.assertEqual(sampler._sampler.enlargement_factor(), x)
        self.assertRaisesRegex(
            ValueError, 'exceed 1',
            sampler._sampler.set_enlargement_factor, 0.5)
        self.assertRaisesRegex(
            ValueError, 'exceed 1',
            sampler._sampler.set_enlargement_factor, 1)

        # Ellipsoid update gap
        x = sampler._sampler.ellipsoid_update_gap() * 2
        self.assertNotEqual(sampler._sampler.ellipsoid_update_gap(), x)
        sampler._sampler.set_ellipsoid_update_gap(x)
        self.assertEqual(sampler._sampler.ellipsoid_update_gap(), x)
        self.assertRaisesRegex(
            ValueError, 'exceed 1',
            sampler._sampler.set_ellipsoid_update_gap, 0.5)
        self.assertRaisesRegex(
            ValueError, 'exceed 1',
            sampler._sampler.set_ellipsoid_update_gap, 1)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
