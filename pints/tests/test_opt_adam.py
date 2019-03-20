#!/usr/bin/env python3
#
# Tests the basic methods of the CMAES optimiser.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import unittest
import numpy as np

import pints
import pints.toy

from shared import StreamCapture, CircularBoundaries

debug = False
method = pints.AdaptiveMomentEstimation

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestAdam(unittest.TestCase):
    """
    Tests the basic methods of the Adam optimiser.
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
        """ Runs an optimisation without boundaries. """
        r, x, s, b = self.problem()
        opt = pints.OptimisationController(r, x, method=method)
        opt.set_threshold(1e-3)
        opt.set_log_to_screen(debug)
        found_parameters, found_solution = opt.run()
        self.assertTrue(found_solution < 1e-3)

    def test_bounded(self):
        """ Runs an optimisation with boundaries. """
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
        """ Runs an optimisation without boundaries and sigma. """
        r, x, s, b = self.problem()
        opt = pints.OptimisationController(r, x, s, b, method)
        opt.set_threshold(1e-3)
        opt.set_log_to_screen(debug)
        found_parameters, found_solution = opt.run()
        self.assertTrue(found_solution < 1e-3)

    def test_ask_tell(self):
        """ Tests ask-and-tell related error handling. """
        r, x, s, b = self.problem()
        opt = method(x)

        # Stop called when not running
        self.assertFalse(opt.stop())

        # Best position and score called before run
        self.assertEqual(list(opt.xbest()), list(x))
        self.assertEqual(opt.fbest(), float('inf'))

        # Tell before ask
        self.assertRaisesRegex(
            Exception, 'ask\(\) not called before tell\(\)', opt.tell, 5)

    def test_hyper_parameter_interface(self):
        """
        Tests the hyper parameter interface for this optimiser.
        """
        r, x, s, b = self.problem()
        opt = pints.OptimisationController(r, x, method=method)
        m = opt.optimiser()
        self.assertEqual(m.n_hyper_parameters(), 3)
        m.set_hyper_parameters([1.0, 0.5, 0.6])
        self.assertEqual(m.alpha(), 1.0)
        self.assertEqual(m.beta1(), 0.5)
        self.assertEqual(m.beta2(), 0.6)
        self.assertRaisesRegex(
            ValueError, 'alpha', m.set_hyper_parameters, [-1.0, 0.5, 0.6])
        self.assertRaisesRegex(
            ValueError, 'beta1', m.set_hyper_parameters, [1.0, 1.5, 0.6])
        self.assertRaisesRegex(
            ValueError, 'beta1', m.set_hyper_parameters, [1.0, -1.5, 0.6])
        self.assertRaisesRegex(
            ValueError, 'beta2', m.set_hyper_parameters, [1.0, 0.5, 1.6])
        self.assertRaisesRegex(
            ValueError, 'beta2', m.set_hyper_parameters, [1.0, 0.5, -0.6])

    def test_name(self):
        """ Test the name() method. """
        opt = method(np.array([0, 1.01]))
        self.assertIn('Adam', opt.name())


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
