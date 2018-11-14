#!/usr/bin/env python
#
# Tests the basic methods of the XNES optimiser.
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

from shared import StreamCapture

debug = False
method = pints.XNES

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestXNES(unittest.TestCase):
    """
    Tests shared optimisation properties.
    """

    def setUp(self):
        """ Called before every test """
        np.random.seed(1)

    def test_optimise(self):
        """ Tests :meth: `pints.optimise()`. """
        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x = np.array([0, 1.01])
        s = 0.01
        b = pints.RectangularBoundaries([-0.01, 0.95], [0.01, 1.05])
        with StreamCapture():
            x, f = pints.optimise(r, x, s, b, method=pints.XNES)
        self.assertEqual(x.shape, (2, ))
        self.assertTrue(f < 1e-6)

    def test_stopping_max_iterations(self):
        """ Runs an optimisation with the max_iter stopping criterion. """
        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x = np.array([0, 1.01])
        b = pints.RectangularBoundaries([-0.01, 0.95], [0.01, 1.05])
        s = 0.01
        opt = pints.Optimisation(r, x, s, b, method)
        opt.set_log_to_screen(True)
        opt.set_max_unchanged_iterations(None)
        opt.set_max_iterations(10)
        self.assertEqual(opt.max_iterations(), 10)
        self.assertRaises(ValueError, opt.set_max_iterations, -1)
        with StreamCapture() as c:
            opt.run()
            self.assertIn('Halting: Maximum number of iterations', c.text())

    def test_logging(self):
        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x = np.array([0, 1.01])
        b = pints.RectangularBoundaries([-0.01, 0.95], [0.01, 1.05])
        s = 0.01
        opt = pints.Optimisation(r, x, s, b, method)
        opt.set_log_to_screen(True)
        opt.set_max_unchanged_iterations(None)
        opt.set_log_interval(3)
        opt.set_max_iterations(10)
        self.assertEqual(opt.max_iterations(), 10)
        with StreamCapture() as c:
            opt.run()

            log_should_be = (
                'Maximising LogPDF\n'
                'using Exponential Natural Evolution Strategy (xNES)\n'
                'Running in sequential mode.\n'
                'Population size: 6\n'
                'Iter. Eval. Best      Time m:s\n'
                '0     6     -1.837877   0:00.0\n'
                '1     12    -1.837877   0:00.0\n'
                '2     18    -1.837877   0:00.0\n'
                '3     24    -1.837877   0:00.0\n'
                '6     42    -1.837877   0:00.0\n'
                '9     60    -1.837877   0:00.0\n'
                '10    60    -1.837877   0:00.0\n'
                'Halting: Maximum number of iterations (10) reached.\n'
            )
            self.assertEqual(log_should_be, c.text())

        # Invalid log interval
        self.assertRaises(ValueError, opt.set_log_interval, 0)

    def test_stopping_max_unchanged(self):
        """ Runs an optimisation with the max_unchanged stopping criterion. """
        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x = np.array([0, 1.01])
        b = pints.RectangularBoundaries([-0.01, 0.95], [0.01, 1.05])
        s = 0.01
        opt = pints.Optimisation(r, x, s, b, method)
        opt.set_log_to_screen(True)
        opt.set_max_iterations(None)
        opt.set_max_unchanged_iterations(None)
        self.assertEqual(opt.max_unchanged_iterations(), (None, None))
        opt.set_max_unchanged_iterations(2, 1e-6)
        self.assertEqual(opt.max_unchanged_iterations(), (2, 1e-6))
        opt.set_max_unchanged_iterations(3)
        self.assertEqual(opt.max_unchanged_iterations(), (3, 1e-11))
        self.assertRaises(ValueError, opt.set_max_unchanged_iterations, -1)
        self.assertRaises(ValueError, opt.set_max_unchanged_iterations, 10, -1)
        with StreamCapture() as c:
            opt.run()
            self.assertIn('Halting: No significant change', c.text())

    def test_stopping_threshold(self):
        """ Runs an optimisation with the threshold stopping criterion. """
        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x = np.array([0.008, 1.01])
        b = pints.RectangularBoundaries([-0.01, 0.95], [0.01, 1.05])
        s = 0.01
        opt = pints.Optimisation(r, x, s, b, method)
        opt.set_log_to_screen(True)
        opt.set_max_iterations(None)
        opt.set_max_unchanged_iterations(None)
        opt.set_threshold(2)
        self.assertEqual(opt.threshold(), 2)
        with StreamCapture() as c:
            opt.run()
            self.assertIn(
                'Halting: Objective function crossed threshold', c.text())

    def test_stopping_no_criterion(self):
        """ Tries to run an optimisation with the no stopping criterion. """
        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x = np.array([0, 1.01])
        b = pints.RectangularBoundaries([-0.01, 0.95], [0.01, 1.05])
        s = 0.01
        opt = pints.Optimisation(r, x, s, b, method)
        opt.set_log_to_screen(debug)
        opt.set_max_iterations(None)
        opt.set_max_unchanged_iterations(None)
        self.assertRaises(ValueError, opt.run)

    def test_set_population_size(self):
        """
        Tests the set_population_size method for this optimiser.
        """
        r = pints.toy.RosenbrockError(1, 100)
        x = np.array([1.01, 1.01])
        opt = pints.Optimisation(r, x, method=method)
        m = opt.optimiser()
        n = m.population_size()
        m.set_population_size(n + 1)
        self.assertEqual(m.population_size(), n + 1)

        # Test invalid size
        self.assertRaisesRegex(
            ValueError, 'at least 1', m.set_population_size, 0)

        # test hyper parameter interface
        self.assertEqual(m.n_hyper_parameters(), 1)
        m.set_hyper_parameters([n + 2])
        self.assertEqual(m.population_size(), n + 2)
        self.assertRaisesRegex(
            ValueError, 'at least 1', m.set_hyper_parameters, [0])

        # Test changing during run
        m.ask()
        self.assertRaises(Exception, m.set_population_size, 2)

    def test_parallel(self):
        """ Test parallelised running. """

        r = pints.toy.RosenbrockError(1, 100)
        x = np.array([1.1, 1.1])
        b = pints.RectangularBoundaries([0.5, 0.5], [1.5, 1.5])

        # Run with guessed number of cores
        opt = pints.Optimisation(r, x, boundaries=b, method=method)
        opt.set_max_iterations(10)
        opt.set_log_to_screen(debug)
        opt.set_parallel(False)
        self.assertIs(opt.parallel(), False)
        opt.set_parallel(True)
        self.assertTrue(type(opt.parallel()) == int)
        self.assertTrue(opt.parallel() >= 1)
        opt.run()

        # Run with explicit number of cores
        opt = pints.Optimisation(r, x, boundaries=b, method=method)
        opt.set_max_iterations(10)
        opt.set_log_to_screen(debug)
        opt.set_parallel(1)
        opt.run()
        self.assertTrue(type(opt.parallel()) == int)
        self.assertEqual(opt.parallel(), 1)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
