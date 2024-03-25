#!/usr/bin/env python3
#
# Tests the basic methods of the Gradient Descent optimiser.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np

import pints
import pints.toy


debug = False
method = pints.GradientDescent


class TestGradientDescent(unittest.TestCase):
    """
    Tests the basic methods of the gradient descent optimiser.

    This method requires gradients & does not respect boundaries.
    """
    def setUp(self):
        """ Called before every test """
        np.random.seed(1)

    def problem(self):
        """ Returns a test problem, starting point and sigma. """
        r = pints.toy.ParabolicError()
        x = [0.1, 0.1]
        s = 0.1
        return r, x, s

    def test_without_sigma(self):
        # Runs an optimisation without sigma.
        r, x, s = self.problem()
        opt = pints.OptimisationController(r, x, method=method)
        opt.set_threshold(1e-3)
        opt.set_log_to_screen(debug)
        found_parameters, found_solution = opt.run()
        self.assertTrue(found_solution < 1e-3)

    def test_with_sigma(self):
        # Runs an optimisation with a sigma.
        r, x, s = self.problem()
        opt = pints.OptimisationController(r, x, s, method=method)
        opt.set_threshold(1e-3)
        opt.set_log_to_screen(debug)
        found_parameters, found_solution = opt.run()
        self.assertTrue(found_solution < 1e-3)

    def test_ask_tell(self):
        # Tests ask-and-tell related error handling.
        r, x, s = self.problem()
        opt = method(x)

        # Stop called when not running
        self.assertFalse(opt.running())
        self.assertFalse(opt.stop())

        # Best position and score called before run
        self.assertEqual(list(opt.x_best()), list(x))
        self.assertEqual(list(opt.x_guessed()), list(x))
        self.assertEqual(opt.f_best(), np.inf)
        self.assertEqual(opt.f_guessed(), np.inf)

        # Tell before ask
        self.assertRaisesRegex(
            Exception, r'ask\(\) not called before tell\(\)', opt.tell, 5)

        # Ask
        opt.ask()

        # Now we should be running
        self.assertTrue(opt.running())

    def test_hyper_parameter_interface(self):
        # Tests the hyper parameter interface for this optimiser.
        r, x, s = self.problem()
        opt = pints.OptimisationController(r, x, method=method)
        m = opt.optimiser()
        self.assertEqual(m.n_hyper_parameters(), 1)
        eta = m.learning_rate() * 2
        m.set_hyper_parameters([eta])
        self.assertEqual(m.learning_rate(), eta)
        self.assertRaisesRegex(
            ValueError, 'greater than zero', m.set_hyper_parameters, [0])

    def test_name(self):
        # Test the name() method.
        opt = method(np.array([0, 1.01]))
        self.assertIn('radient descent', opt.name())

    def test_step(self):
        # Numerically test that it takes the correct step

        # Create a gradient descent optimiser starting at 0
        x0 = np.zeros(8)
        opt = pints.GradientDescent(x0)
        opt.set_learning_rate(1)

        # Check that it starts at 0
        xs = opt.ask()
        self.assertEqual(len(xs), 1)
        # Cast to list gives nicest message if any elements don't match
        self.assertEqual(list(xs[0]), list(x0))

        # If we pass in gradient -g, we should move to g
        g = np.array([1, 2, 3, 4, 8, -7, 6, 5])
        opt.tell([(0, -g)])
        ys = opt.ask()
        self.assertEqual(list(ys[0]), list(g))

        # If we halve the learning rate and pass in +g, we should retrace half
        # a step
        opt.set_learning_rate(0.5)
        opt.tell([(0, g)])
        ys = opt.ask()
        self.assertEqual(list(ys[0]), list(0.5 * g))

        # And if we pass in +g again we should be back at 0
        opt.tell([(0, g)])
        ys = opt.ask()
        self.assertEqual(list(ys[0]), list(x0))


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()

