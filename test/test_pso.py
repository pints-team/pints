#!/usr/bin/env python
#
# Tests the basic methods of the PSO optimiser.
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

debug = False
method = pints.PSO

LOG_SCREEN = (
    'Maximising LogPDF\n'
    'using Particle Swarm Optimisation (PSO)\n'
    'Running in parallel with 2 worker processes.\n'
    'Population size: 6\n'
    'Iter. Eval. Best      Time m:s\n'
    '0     6     -0.199      0:00.0\n'
    '1     12     0.843      0:00.0\n'
    '2     18     0.843      0:00.0\n'
    '3     24     0.843      0:00.0\n'
    '10    60     1.183198   0:00.0\n'
    'Halting: Maximum number of iterations (10) reached.\n'
)

LOG_FILE = (
    'Iter. Eval. Best      f0        f1        f2        f3       '
    ' f4        f5        Time m:s\n'
    '0     6     -0.199     0.199     2.667294  3.425707  1.148659  2.648408 '
    ' 1.707569   0:00.0\n'
    '1     12     0.843     0.199    -0.843     1.621933  1.148659  1.283276 '
    ' 1.487763   0:00.0\n'
    '2     18     0.843    -0.196    -0.843     1.621933  1.148659  1.283276 '
    ' 1.487763   0:00.0\n'
    '3     24     0.843    -0.196    -0.843     0.631     1.148659 -0.404    '
    ' 1.487763   0:00.0\n'
    '10    60     1.183198 -1.183198 -0.843     0.631     1.148659 -0.404    '
    '-0.0216     0:00.0\n'
)


class TestPSO(unittest.TestCase):
    """
    Tests the basic methods of the PSO optimiser.
    """
    def __init__(self, name):
        super(TestPSO, self).__init__(name)

        # Create toy model
        self.model = pints.toy.LogisticModel()
        self.real_parameters = [0.015, 500]
        self.times = np.linspace(0, 1000, 1000)
        self.values = self.model.simulate(self.real_parameters, self.times)

        # Create an object with links to the model and time series
        self.problem = pints.SingleOutputProblem(
            self.model, self.times, self.values)

        # Select a score function
        self.score = pints.SumOfSquaresError(self.problem)

        # Select some boundaries
        self.boundaries = pints.Boundaries([0, 400], [0.03, 600])

        # Set an initial position
        self.x0 = 0.014, 499

        # Set an initial guess of the standard deviation in each parameter
        self.sigma0 = [0.001, 1]

        # Minimum score function value to obtain
        self.cutoff = 1e3   # Global method!

        # Maximum tries before it counts as failed
        self.max_tries = 3

    def test_bounded(self):

        opt = pints.Optimisation(self.score, self.x0,
                                 boundaries=self.boundaries, method=method)
        opt.set_log_to_screen(debug)
        opt.set_max_unchanged_iterations(1000)
        for i in range(self.max_tries):
            found_parameters, found_solution = opt.run()
            if found_solution < self.cutoff:
                break
        self.assertTrue(found_solution < self.cutoff)

    def test_bounded_and_sigma(self):

        opt = pints.Optimisation(self.score, self.x0, self.sigma0,
                                 self.boundaries, method)
        opt.set_log_to_screen(debug)
        opt.set_max_unchanged_iterations(1000)
        for i in range(self.max_tries):
            found_parameters, found_solution = opt.run()
            if found_solution < self.cutoff:
                break
        self.assertTrue(found_solution < self.cutoff)

    def test_stopping_max_iter(self):

        opt = pints.Optimisation(self.score, self.x0, self.sigma0,
                                 self.boundaries, method)
        opt.set_log_to_screen(True)
        opt.set_max_iterations(2)
        opt.set_max_unchanged_iterations(None)
        with StreamCapture() as c:
            opt.run()
            self.assertIn('Halting: Maximum number of iterations', c.text())

    def test_stopping_max_unchanged(self):

        opt = pints.Optimisation(self.score, self.x0, self.sigma0,
                                 self.boundaries, method)
        opt.set_log_to_screen(True)
        opt.set_max_iterations(None)
        opt.set_max_unchanged_iterations(2)
        self.assertEqual(opt.max_unchanged_iterations()[0], 2)
        with StreamCapture() as c:
            opt.run()
            self.assertIn('Halting: No significant change', c.text())

    def test_stopping_threshold(self):

        opt = pints.Optimisation(self.score, self.x0, self.sigma0,
                                 self.boundaries, method)
        opt.set_log_to_screen(True)
        opt.set_max_iterations(None)
        opt.set_max_unchanged_iterations(None)
        self.assertEqual(opt.max_unchanged_iterations(), (None, None))
        t = 1e4 * self.cutoff
        opt.set_threshold(t)
        self.assertEqual(opt.threshold(), t)
        with StreamCapture() as c:
            opt.run()
            self.assertIn(
                'Halting: Objective function crossed threshold', c.text())

    def test_stopping_no_criterion(self):

        opt = pints.Optimisation(self.score, self.x0, self.sigma0,
                                 self.boundaries, method)
        opt.set_log_to_screen(debug)
        opt.set_max_iterations(None)
        opt.set_max_unchanged_iterations(None)
        self.assertEqual(opt.max_unchanged_iterations(), (None, None))
        self.assertRaises(ValueError, opt.run)

    def test_logpdf(self):

        r = pints.toy.RosenbrockLogPDF(1, 100)
        x0 = np.array([1.1, 1.1])
        f0 = r(x0)
        b = pints.Boundaries([0.5, 0.5], [1.5, 1.5])
        opt = pints.Optimisation(r, x0, boundaries=b, method=method)
        opt.set_max_iterations(100)
        opt.set_max_unchanged_iterations(100)
        opt.set_log_to_screen(debug)

        # PSO isn't very good at this function, unless we give it lots more
        # iterations, check if it's moving in the right direction
        np.random.seed(1)
        x1, f1 = opt.run()
        self.assertTrue(f1 > f0)

    def test_rosenbrock(self):
        """ Test running on the Rosenbrock function """

        r = pints.toy.RosenbrockError(1, 100)
        x0 = np.array([1.1, 1.1])
        f0 = r(x0)
        b = pints.Boundaries([0.5, 0.5], [1.5, 1.5])
        opt = pints.Optimisation(r, x0, boundaries=b, method=method)
        opt.set_max_iterations(100)
        opt.set_max_unchanged_iterations(100)
        opt.set_log_to_screen(debug)
        np.random.seed(1)
        x1, f1 = opt.run()
        self.assertTrue(f1 < f0)

    def test_logging(self):

        r = pints.toy.RosenbrockLogPDF(1, 100)
        x0 = np.array([1.1, 1.1])
        b = pints.Boundaries([0.5, 0.5], [1.5, 1.5])

        # No logging
        with StreamCapture() as c:
            opt = pints.Optimisation(r, x0, boundaries=b, method=method)
            opt.set_max_iterations(10)
            opt.set_log_to_screen(False)
            np.random.seed(1)
            opt.run()
        self.assertEqual(c.text(), '')

        # Log to screen
        with StreamCapture() as c:
            opt = pints.Optimisation(r, x0, boundaries=b, method=method)
            opt.set_max_iterations(10)
            opt.set_parallel(2)
            opt.set_log_to_screen(True)
            np.random.seed(1)
            opt.run()
        self.assertEqual(c.text(), LOG_SCREEN)

        # Log to file
        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                filename = d.path('test.txt')
                opt = pints.Optimisation(r, x0, boundaries=b, method=method)
                opt.set_max_iterations(10)
                opt.set_log_to_screen(False)
                opt.set_log_to_file(filename)
                np.random.seed(1)
                opt.run()
                with open(filename, 'r') as f:
                    self.assertEqual(f.read(), LOG_FILE)
            self.assertEqual(c.text(), '')

    def test_parallel(self):
        """ Test parallelised running on the Rosenbrock function. """

        r = pints.toy.RosenbrockError(1, 100)
        x0 = np.array([1.1, 1.1])
        b = pints.Boundaries([0.5, 0.5], [1.5, 1.5])

        # Run with guessed number of cores
        opt = pints.Optimisation(r, x0, boundaries=b, method=method)
        opt.set_max_iterations(10)
        opt.set_log_to_screen(debug)
        opt.set_parallel(False)
        self.assertIs(opt.parallel(), False)
        opt.set_parallel(True)
        self.assertTrue(type(opt.parallel()) == int)
        self.assertTrue(opt.parallel() >= 1)
        opt.run()

        # Run with explicit number of cores
        opt = pints.Optimisation(r, x0, boundaries=b, method=method)
        opt.set_max_iterations(10)
        opt.set_log_to_screen(debug)
        opt.set_parallel(1)
        opt.run()
        self.assertTrue(type(opt.parallel()) == int)
        self.assertEqual(opt.parallel(), 1)

    def test_suggest_population_size(self):
        """
        Tests the suggested_population_size() method for population based
        optimisers.
        """
        r = pints.toy.RosenbrockError(1, 100)
        x0 = np.array([1.1, 1.1])
        b = pints.Boundaries([0.5, 0.5], [1.5, 1.5])
        opt = pints.Optimisation(r, x0, boundaries=b, method=method)
        opt = opt.optimiser()

        # Test basic usage
        self.assertEqual(type(opt.suggested_population_size()), int)
        self.assertTrue(opt.suggested_population_size() > 0)

        # Test rounding
        n = opt.suggested_population_size() + 1
        self.assertEquals(opt.suggested_population_size(n), n)
        self.assertEquals(opt.suggested_population_size(2) % 2, 0)
        self.assertEquals(opt.suggested_population_size(3) % 3, 0)
        self.assertEquals(opt.suggested_population_size(5) % 5, 0)
        self.assertEquals(opt.suggested_population_size(7) % 7, 0)
        self.assertEquals(opt.suggested_population_size(11) % 11, 0)

    def test_creation(self):
        """ Test optimiser creation. """
        # Test basic creation
        x0 = [1, 2, 3]
        pints.PSO(x0)
        self.assertRaisesRegexp(
            ValueError, 'greater than zero', pints.PSO, [])

        # Test with boundaries
        x0 = [1, 2]
        b = pints.Boundaries([0, 0], [3, 3])
        pints.PSO(x0, boundaries=b)
        self.assertRaisesRegexp(
            ValueError, 'within given boundaries', pints.PSO, [4, 4],
            boundaries=b)

        # Test with scalar sigma
        pints.PSO(x0, 3)
        self.assertRaisesRegexp(
            ValueError, 'greater than zero', pints.PSO, x0, -1)

        # Test with vector sigma
        pints.PSO(x0, [3, 3])
        self.assertRaisesRegexp(
            ValueError, 'greater than zero', pints.PSO, x0, [3, -1])
        self.assertRaisesRegexp(
            ValueError, 'have dimension 2', pints.PSO, x0, [3, 3, 3])

    def test_optimisation_creation(self):
        """
        Test optimisation creation.
        """
        # Test invalid dimensions
        r = pints.toy.RosenbrockError(1, 100)
        self.assertRaisesRegexp(
            ValueError, 'same dimension', pints.Optimisation, r, [1])

        # Test invalid method
        self.assertRaisesRegexp(
            ValueError, 'subclass', pints.Optimisation, r, [1, 1],
            method=pints.AdaptiveCovarianceMCMC)

    def test_set_population_size(self):
        """
        Tests the set_population_size method for this optimiser.
        """
        r = pints.toy.RosenbrockError(1, 100)
        x0 = np.array([1.1, 1.1])
        b = pints.Boundaries([0.5, 0.5], [1.5, 1.5])
        opt = pints.Optimisation(r, x0, boundaries=b, method=method)
        m = opt.optimiser()
        n = m.population_size()
        m.set_population_size(n + 1)
        self.assertEqual(m.population_size(), n + 1)

        # Test invalid size
        self.assertRaisesRegexp(
            ValueError, 'at least 1', m.set_population_size, 0)

        # Test changing during run
        m.ask()
        self.assertRaises(Exception, m.set_population_size, 2)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
