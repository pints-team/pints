#!/usr/bin/env python3
#
# Tests the basic methods of the bare-bones CMAES optimiser.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np

import pints
import pints.toy

from shared import CircularBoundaries

debug = False
method = pints.BareCMAES

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestBareCMAES(unittest.TestCase):
    """
    Tests the basic methods of the BareCMAES optimiser.
    """
    def setUp(self):
        """ Called before every test """
        np.random.seed(3)

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
        opt.set_threshold(1e-3)
        opt.set_log_to_screen(debug)
        found_parameters, found_solution = opt.run()
        self.assertTrue(found_solution < 1e-3)

    def test_bounded(self):
        # Runs an optimisation with boundaries.
        r, x, s, b = self.problem()

        # Rectangular boundaries
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
        opt = pints.OptimisationController(r, x, s, b, method)
        opt.set_threshold(1e-3)
        opt.set_log_to_screen(debug)
        found_parameters, found_solution = opt.run()
        self.assertTrue(found_solution < 1e-3)

    def test_ask_tell(self):
        # Tests ask-and-tell related error handling.
        r, x, s, b = self.problem()
        opt = method(x)

        # Stop called when not running
        self.assertFalse(opt.stop())

        # Best position and score called before run
        self.assertEqual(list(opt.xbest()), list(x))
        self.assertEqual(opt.fbest(), float('inf'))

        # Tell before ask
        self.assertRaisesRegex(
            Exception, r'ask\(\) not called before tell\(\)', opt.tell, 5)

    def test_hyper_parameter_interface(self):
        # Tests the hyper parameter interface for this optimiser.
        r, x, s, b = self.problem()
        opt = pints.OptimisationController(r, x, method=method)
        m = opt.optimiser()
        self.assertEqual(m.n_hyper_parameters(), 1)
        n = m.population_size() + 2
        m.set_hyper_parameters([n])
        self.assertEqual(m.population_size(), n)
        self.assertRaisesRegex(
            ValueError, 'at least 1', m.set_hyper_parameters, [0])

    def test_name(self):
        # Test the name() method.
        opt = method(np.array([0, 1.01]))
        self.assertIn('Bare-bones CMA-ES', opt.name())

    def test_covariance_matrix_and_mean(self):
        # Tests getting the covariance matrix and mean

        r, x, s, b = self.problem()
        opt = method(x)

        e = pints.ParallelEvaluator(r)
        for i in range(10):
            opt.tell(e.evaluate(opt.ask()))

        # Get covariance matrix: check shape and symmetry
        C = opt.cov()
        self.assertEqual(C.shape, (2, 2))
        self.assertTrue(np.all(C == C.T))

        # Get decomposition
        R, S = opt.cov(decomposed=True)
        error = np.max(np.abs(C - R.dot(S).dot(S).dot(R.T)))
        self.assertLess(error, 1e-15)

        # Get mean
        mean = opt.mean()
        self.assertEqual(mean.shape, (2, ))


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
        import logging
        logging.basicConfig(level=logging.DEBUG)
    unittest.main()
