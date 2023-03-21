#!/usr/bin/env python3
#
# Tests the basic methods of the PSO optimiser.
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

from shared import StreamCapture, TemporaryDirectory, CircularBoundaries


debug = False
method = pints.PSO


class TestPSO(unittest.TestCase):
    """
    Tests the basic methods of the PSO optimiser.
    """

    def setUp(self):
        """ Called before every test """
        np.random.seed(1)

    def problem(self):
        """ Returns a test problem, starting point, sigma, and boundaries. """
        r = pints.toy.ParabolicError()
        x = [0.1, 0.1]
        s = 0.1
        b = pints.RectangularBoundaries([-1, -1], [1, 1])
        return r, x, s, b

    def test_unbounded(self):
        # Runs an optimisation without boundaries.
        r, x, s, b = self.problem()
        opt = pints.OptimisationController(r, x, method=method)
        opt.set_log_to_screen(debug)
        found_parameters, found_solution = opt.run()
        self.assertTrue(found_solution < 1e-3)

    def test_bounded(self):
        # Runs an optimisation with boundaries.
        r, x, s, b = self.problem()

        # Rectangular boundaries
        b = pints.RectangularBoundaries([-1, -1], [1, 1])
        opt = pints.OptimisationController(r, x, boundaries=b, method=method)
        opt.set_log_to_screen(debug)
        found_parameters, found_solution = opt.run()
        self.assertTrue(found_solution < 1e-3)

        # Circular boundaries
        # Start near edge, to increase chance of out-of-bounds occurring.
        b = CircularBoundaries([0, 0], 1)
        x = [0.99, 0]
        opt = pints.OptimisationController(r, x, boundaries=b, method=method)
        opt.set_log_to_screen(debug)
        found_parameters, found_solution = opt.run()
        self.assertTrue(found_solution < 1e-3)

    def test_bounded_and_sigma(self):
        # Runs an optimisation without boundaries and sigma.
        r, x, s, b = self.problem()
        opt = pints.OptimisationController(r, x, s, b, method=method)
        opt.set_log_to_screen(debug)
        #found_parameters, found_solution = opt.run()
        #self.assertTrue(found_solution < 1e-3)

    def test_ask_tell(self):
        # Tests ask-and-tell related error handling.
        r, x, s, b = self.problem()
        opt = method(x)

        # Stop called when not running
        self.assertFalse(opt.stop())

        # Best position and score called before run
        self.assertEqual(list(opt.x_best()), list(x))
        self.assertEqual(list(opt.x_guessed()), list(x))
        self.assertEqual(opt.f_best(), np.inf)
        self.assertEqual(opt.f_guessed(), np.inf)

        # Tell before ask
        self.assertRaisesRegex(
            Exception, r'ask\(\) not called before tell\(\)', opt.tell, 5)

        # Can't change settings while running
        opt.ask()
        self.assertRaisesRegex(
            Exception, 'during run', opt.set_local_global_balance, 0.1)

    def test_logging(self):
        # Tests logging for PSO and other optimisers.
        # Use a LogPDF to test if it shows the maximising message!
        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x = np.array([0, 1.01])
        b = pints.RectangularBoundaries([-0.01, 0.95], [0.01, 1.05])
        s = 0.01
        opt = pints.OptimisationController(r, x, s, b, method=method)

        # No logging
        opt = pints.OptimisationController(r, x, s, b, method=method)
        opt.set_max_iterations(10)
        opt.set_log_to_screen(False)
        opt.set_log_to_file(False)
        with StreamCapture() as c:
            opt.run()
        self.assertEqual(c.text(), '')

        # Log to screen - using a LogPDF and parallelisation
        opt = pints.OptimisationController(r, x, s, b, method=method)
        opt.set_parallel(4)
        opt.set_max_iterations(10)
        opt.set_log_to_screen(True)
        opt.set_log_to_file(False)
        with StreamCapture() as c:
            opt.run()
        lines = c.text().splitlines()

        self.assertEqual(len(lines), 11)
        self.assertEqual(lines[0], 'Maximising LogPDF')
        self.assertEqual(lines[1], 'Using Particle Swarm Optimisation (PSO)')
        self.assertEqual(
            lines[2], 'Running in parallel with 4 worker processes.')
        self.assertEqual(lines[3], 'Population size: 6')
        self.assertEqual(lines[4], 'Iter. Eval. Best      Current   Time m:s')

        pint = '[0-9]+[ ]+'
        pflt = '[0-9.-]+[ ]+'
        ptim = '[0-9]{1}:[0-9]{2}.[0-9]{1}'
        pattern = re.compile(pint * 2 + 2 * pflt + ptim)
        for line in lines[5:-1]:
            self.assertTrue(pattern.match(line))
        self.assertEqual(
            lines[-1], 'Halting: Maximum number of iterations (10) reached.')

        # Log to file
        opt = pints.OptimisationController(r, x, s, b, method=method)
        opt.set_max_iterations(10)
        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                filename = d.path('test.txt')
                opt.set_log_to_screen(False)
                opt.set_log_to_file(filename)
                opt.run()
                with open(filename, 'r') as f:
                    lines = f.read().splitlines()
            self.assertEqual(c.text(), '')

        self.assertEqual(len(lines), 6)
        self.assertEqual(
            lines[0],
            'Iter. Eval. Best      Current   f0        f1        f2        '
            'f3        f4        f5        Time m:s'
        )

        pattern = re.compile(pint * 2 + pflt * 8 + ptim)
        for line in lines[1:]:
            self.assertTrue(pattern.match(line))

    def test_suggest_population_size(self):
        # Tests the suggested_population_size() method for population based
        # optimisers.

        r, x, s, b = self.problem()
        opt = pints.OptimisationController(r, x, boundaries=b, method=method)
        opt = opt.optimiser()

        # Test basic usage
        self.assertEqual(type(opt.suggested_population_size()), int)
        self.assertTrue(opt.suggested_population_size() > 0)

        # Test rounding
        n = opt.suggested_population_size() + 1
        self.assertEqual(opt.suggested_population_size(n), n)
        self.assertEqual(opt.suggested_population_size(2) % 2, 0)
        self.assertEqual(opt.suggested_population_size(3) % 3, 0)
        self.assertEqual(opt.suggested_population_size(5) % 5, 0)
        self.assertEqual(opt.suggested_population_size(7) % 7, 0)
        self.assertEqual(opt.suggested_population_size(11) % 11, 0)

    def test_creation(self):

        # Test basic creation
        r, x, s, b = self.problem()
        method(x)
        self.assertRaisesRegex(
            ValueError, 'greater than zero', pints.PSO, [])

        # Test with boundaries
        method(x, boundaries=b)
        self.assertRaisesRegex(
            ValueError, 'within given boundaries', method, [4, 4],
            boundaries=b)
        self.assertRaisesRegex(
            ValueError, 'same dimension', method, [0, 0, 0],
            boundaries=b)

        # Test with scalar sigma
        method(x, 3)
        self.assertRaisesRegex(
            ValueError, 'greater than zero', method, x, -1)

        # Test with vector sigma
        method(x, [3, 3])
        self.assertRaisesRegex(
            ValueError, 'greater than zero', method, x, [3, -1])
        self.assertRaisesRegex(
            ValueError, 'have dimension 2', method, x, [3, 3, 3])

    def test_controller_creation(self):

        # Test invalid dimensions
        r, x, s, b = self.problem()
        self.assertRaisesRegex(
            ValueError, 'same dimension', pints.OptimisationController, r, [1])

        # Test invalid method
        self.assertRaisesRegex(
            ValueError, 'subclass', pints.OptimisationController, r, [1, 1],
            method=pints.HaarioBardenetACMC)

    def test_set_hyper_parameters(self):
        # Tests the hyper-parameter interface for this optimiser.

        r, x, s, b = self.problem()
        opt = pints.OptimisationController(r, x, boundaries=b, method=method)
        m = opt.optimiser()
        self.assertEqual(m.n_hyper_parameters(), 2)
        n = m.population_size()

        m.set_hyper_parameters([n + 1, 0.5])
        self.assertEqual(m.population_size(), n + 1)

        # Test invalid size
        self.assertRaisesRegex(
            ValueError, 'at least 1', m.set_hyper_parameters, [0, 0.5])
        self.assertRaisesRegex(
            ValueError, 'in the range 0-1', m.set_hyper_parameters, [n, 1.5])

    def test_name(self):
        # Test the name() method.
        opt = method(np.array([0, 1.01]))
        self.assertIn('PSO', opt.name())


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
