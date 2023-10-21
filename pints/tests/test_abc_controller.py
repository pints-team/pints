#!/usr/bin/env python3
#
# Tests the ABC Controller.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.toy
import pints.toy.stochastic
import unittest
import numpy as np

from shared import StreamCapture, TemporaryDirectory


LOG_SCREEN_1 = [
    'Using Rejection ABC',
    'Running in sequential mode.',
    'Iter. Eval. Acceptance rate Time m:s',
    '1     127    0.00787401575    0:00.0',
    '2     167    0.0119760479     0:00.0',
    '3     209    0.014354067      0:00.0',
    'Halting: Target number of samples (3) reached.',
]

LOG_SCREEN_2 = [
    'Using Rejection ABC',
    'Running in parallel with 2 worker processess.',
    'Iter. Eval. Acceptance rate Time m:s',
    '1     20     0.05             0:00.0',
    '2     22     0.0909090909     0:00.0',
    '3     144    0.0208333333     0:00.0',
    '6     308    0.0194805195     0:00.0',
    'Halting: Maximum number of iterations (6) reached. Only 6 samples were'
    ' obtained.',
]

LOG_FILE = [
    'Iter. Eval. Acceptance rate Time m:s',
    '1     127    0.00787401575    0:00.0',
    '2     167    0.0119760479     0:00.0',
    '3     209    0.014354067      0:00.0',
    '6     369    0.0162601626     0:00.0',
]

LOG_FILE_CSV = [
    '"Iter.","Eval.","Acceptance rate","Time m:s"',
    '1,127,7.87401574803149595e-03,0:00.0',
    '2,167,0.0119760479,0:00.0',
    '3,209,0.014354067,0:00.0',
    '6,369,0.0162601626,0:00.0',
]


class TestABCController(unittest.TestCase):
    """
    Tests the ABCController class.
    """

    @classmethod
    def setUpClass(cls):
        """ Prepare problem for tests. """

        # Create toy model
        np.random.seed(1)
        cls.model = pints.toy.stochastic.DegradationModel()
        cls.real_parameters = [0.1]
        cls.times = np.linspace(0, 10, 10)
        cls.values = cls.model.simulate(cls.real_parameters, cls.times)

        # Create an object (problem) with links to the model and time series
        cls.problem = pints.SingleOutputProblem(
            cls.model, cls.times, cls.values)

        # Create a uniform prior over both the parameters
        cls.log_prior = pints.UniformLogPrior(
            [0.0],
            [0.3]
        )

        # Set error measure
        cls.error_measure = pints.RootMeanSquaredError(cls.problem)

    def test_nparameters_error(self):
        # Test that error is thrown when parameters from log prior and error
        # measure do not match.
        log_prior = pints.UniformLogPrior(
            [0.0, 0, 0],
            [0.2, 100, 1])

        self.assertRaises(ValueError, pints.ABCController, self.error_measure,
                          log_prior)

    def test_error_measure_instance(self):
        # Test that error is thrown when we use an error measure which is not
        # an instance of ``pints.ErrorMeasure``.
        # Set a log prior as the error measure to trigger the warning
        wrong_error_measure = pints.UniformLogPrior(
            [0.0, 0, 0],
            [0.2, 100, 1])

        self.assertRaises(
            ValueError,
            pints.ABCController,
            wrong_error_measure,
            self.log_prior)

    def test_stopping(self):
        #" Test different stopping criteria.

        abc = pints.ABCController(self.error_measure, self.log_prior)

        # Test setting max iterations
        maxi = abc.max_iterations() + 2
        self.assertNotEqual(maxi, abc.max_iterations())
        abc.set_max_iterations(maxi)
        self.assertEqual(maxi, abc.max_iterations())
        self.assertRaisesRegex(
            ValueError,
            'Maximum number of iterations cannot be negative.',
            abc.set_max_iterations, -1)

        # # Test without stopping criteria
        abc.set_max_iterations(None)
        self.assertIsNone(abc.max_iterations())
        self.assertRaisesRegex(
            ValueError,
            'At least one stopping criterion must be set.',
            abc.run)

    def test_parallel(self):
        # Test running ABC with parallisation.

        abc = pints.ABCController(
            self.error_measure, self.log_prior, method=pints.RejectionABC)

        # Test with auto-detected number of worker processes
        self.assertFalse(abc.parallel())
        abc.set_parallel(True)
        self.assertTrue(abc.parallel())
        self.assertEqual(abc.parallel(), pints.ParallelEvaluator.cpu_count())

        # Test with fixed number of worker processes
        abc.set_parallel(2)
        self.assertEqual(abc.parallel(), 2)

        abc.sampler().set_threshold(10)
        vals = abc.sampler().ask(3)
        res = abc.sampler().tell([1, 2, 3])

        self.assertCountEqual(vals, res)

    def test_logging(self):
        # Tests logging to screen

        # No output
        abc = pints.ABCController(
            self.error_measure, self.log_prior, method=pints.RejectionABC)
        abc.set_max_iterations(10)
        abc.set_n_samples(3)
        abc.set_log_to_screen(False)
        self.assertIsNone(abc.log_filename())
        abc.set_log_to_file(False)
        self.assertIsNone(abc.log_filename())
        with StreamCapture() as capture:
            abc.run()
        self.assertEqual(capture.text(), '')

        # With output to screen
        np.random.seed(1)
        abc = pints.ABCController(
            self.error_measure, self.log_prior, method=pints.RejectionABC)
        abc.set_max_iterations(10)
        abc.set_n_samples(3)
        abc.set_log_to_screen(True)
        with StreamCapture() as capture:
            abc.run()
        lines = capture.text().splitlines()
        self.assertTrue(len(lines) > 0)
        for i, line in enumerate(lines):
            self.assertLess(i, len(LOG_SCREEN_1))
            # Chop off time bit before comparison
            if LOG_SCREEN_1[i][-6:] == '0:00.0':
                self.assertEqual(line[:-6], LOG_SCREEN_1[i][:-6])
            else:
                self.assertEqual(line, LOG_SCREEN_1[i])
        self.assertEqual(len(lines), len(LOG_SCREEN_1))

        # With output to screen: in parallel and with other stopping crit
        np.random.seed(1)
        abc = pints.ABCController(
            self.error_measure, self.log_prior, method=pints.RejectionABC)
        abc.set_max_iterations(6)
        abc.set_log_interval(6, 3)
        abc.set_parallel(2)
        abc.set_log_to_screen(True)
        with StreamCapture() as capture:
            abc.run()
        lines = capture.text().splitlines()
        self.assertTrue(len(lines) > 0)
        for i, line in enumerate(lines):
            self.assertLess(i, len(LOG_SCREEN_2))
            # Chop off time bit before comparison
            if LOG_SCREEN_2[i][-6:] == '0:00.0':
                self.assertEqual(line[:-6], LOG_SCREEN_2[i][:-6])
            else:
                self.assertEqual(line, LOG_SCREEN_2[i])
        self.assertEqual(len(lines), len(LOG_SCREEN_2))

        # With output to file
        np.random.seed(1)
        with TemporaryDirectory() as d:
            filename = d.path('test.txt')
            abc = pints.ABCController(
                self.error_measure,
                self.log_prior,
                method=pints.RejectionABC)
            abc.set_max_iterations(6)
            abc.set_log_interval(6, 3)
            abc.set_log_to_screen(False)
            abc.set_log_to_file(filename)
            self.assertEqual(abc.log_filename(), filename)

            with StreamCapture() as capture:
                abc.run()
            self.assertEqual(capture.text(), '')

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

        # With output to CSV file
        np.random.seed(1)
        with TemporaryDirectory() as d:
            filename = d.path('test.txt')
            abc = pints.ABCController(
                self.error_measure, self.log_prior, method=pints.RejectionABC)
            abc.set_max_iterations(6)
            abc.set_log_interval(6, 3)
            abc.set_log_to_screen(False)
            abc.set_log_to_file(filename, csv=True)
            self.assertEqual(abc.log_filename(), filename)

            with StreamCapture() as capture:
                abc.run()
            self.assertEqual(capture.text(), '')

            with open(filename, 'r') as f:
                lines = f.read().splitlines()

            for line1, line2 in zip(lines, LOG_FILE_CSV):
                if line2.startswith('"'):
                    self.assertEqual(line1, line2)
                else:
                    parts1 = line1.split(',')
                    parts2 = line2.split(',')
                    self.assertEqual(len(parts1), len(parts2))
                    self.assertEqual(parts1[0], parts2[0])
                    self.assertEqual(parts1[1], parts2[1])
                    self.assertAlmostEqual(float(parts1[2]), float(parts2[2]))
            self.assertEqual(len(lines), len(LOG_FILE_CSV))

        # Invalid log interval
        self.assertRaises(ValueError, abc.set_log_interval, 0)

    def test_controller_extra(self):
        # Tests various controller aspects

        self.assertRaises(ValueError, pints.ABCController, self.error_measure,
                          self.error_measure)
        self.assertRaisesRegex(
            ValueError, 'Given method must extend ABCSampler.',
            pints.ABCController, self.error_measure,
            self.log_prior, pints.MCMCSampler)
        self.assertRaises(ValueError, pints.ABCController, self.error_measure,
                          pints.MCMCSampler)
        self.assertRaises(ValueError, pints.ABCController, self.error_measure,
                          self.log_prior, 0.0)

        # test setters
        abc = pints.ABCController(
            self.error_measure, self.log_prior, method=pints.RejectionABC)
        abc.set_n_samples(230)
        self.assertEqual(abc.n_samples(), 230)

        sampler = abc.sampler()
        pt = sampler.ask(1)
        self.assertEqual(len(pt), 1)

        abc.set_parallel(False)
        self.assertEqual(abc.parallel(), 0)

        abc = pints.ABCController(
            self.error_measure, self.log_prior, method=pints.RejectionABC)
        abc.set_parallel(4)
        abc.sampler().set_threshold(100)
        abc.set_n_samples(1)
        with StreamCapture() as capture:
            abc.run()
        lines = capture.text().splitlines()
        self.assertTrue(len(lines) > 0)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
