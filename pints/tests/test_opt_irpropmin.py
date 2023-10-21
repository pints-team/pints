#!/usr/bin/env python3
#
# Tests the API of the iRprop- optimiser.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np

import pints
import pints.toy

from shared import StreamCapture


debug = False
method = pints.IRPropMin


log_1 = '''
Minimising error measure
Using iRprop-
Running in sequential mode.
Iter. Eval. Best      Current   Min. step Max. step Bound corr. Time m:s
0     1      0.02      0.02      0.12      0.12                   0:00.0
1     2      0.0008    0.0008    0.06      0.06                   0:00.0
'''.strip()

log_2 = '''
Minimising error measure
Using iRprop-
Running in sequential mode.
Iter. Eval. Best      Current   Min. step Max. step Bound corr. Time m:s
0     1      0.02      0.02      0.11      0.11                   0:00.0
1     2      0.0002    0.0002    0.055     0.055                  0:00.0
2     3      0.0002    0.0002    0.055     0.055                  0:00.0
3     4      0.0002    0.00405   0.03      0.03                   0:00.0
'''.strip()


class TestIRPropMin(unittest.TestCase):
    """
    Tests the API of the iRprop- optimiser.
    """

    def problem(self):
        """ Returns a test problem, starting point, and sigma. """
        r = pints.toy.ParabolicError()
        x = [0.1, 0.1]
        s = 0.1
        return r, x, s

    def test_simple(self):
        # Runs an optimisation
        r, x, s = self.problem()

        opt = pints.OptimisationController(r, x, sigma0=s, method=method)
        opt.set_log_to_screen(debug)
        found_parameters, found_solution = opt.run()

        # True solution is (0, 0) with error 0
        self.assertTrue(found_solution < 1e-9)
        self.assertLess(abs(found_parameters[0]), 1e-4)
        self.assertLess(abs(found_parameters[1]), 1e-4)

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

    def test_boundaries(self):
        # Tests boundary support

        # First, set up a test case in which boundaries are breached
        error = pints.toy.ParabolicError()
        x0 = [1.8, -0.8]
        b1 = pints.RectangularBoundaries([-0.1, -0.9], [1.9, 0.9])

        # Test the test: Check that 5 steps is enough to get a boundary breach
        checks = []
        opt = pints.IRPropMin(x0, 1.2)
        for i in range(5):
            xs = opt.ask()
            fs = [error.evaluateS1(x) for x in xs]
            opt.tell(fs)
            checks.append(b1.check(xs[0]))
        self.assertEqual(checks, [True, True, False, False, False])

        # Test with rectangular boundaries
        checks = []
        opt = pints.IRPropMin(x0, 1.2, boundaries=b1)
        for i in range(5):
            xs = opt.ask()
            fs = [error.evaluateS1(x) for x in xs]
            opt.tell(fs)
            checks.append(b1.check(xs[0]))
        self.assertEqual(checks, [True, True, True, True, True])

        # Test with custom boundaries
        class CustomBoundaries(pints.Boundaries):
            lo = np.array([-0.1, -0.9])
            up = np.array([1.9, 0.9])

            def n_parameters(self):
                return 2

            def check(self, x):
                return not np.any((x < self.lo) | (x >= self.up))

        b2 = CustomBoundaries()
        checks1 = []
        checks2 = []
        opt = pints.IRPropMin(x0, 1.2, boundaries=b2)
        for i in range(5):
            xs = opt.ask()
            fs = [error.evaluateS1(x) for x in xs]
            opt.tell(fs)
            checks1.append(b1.check(xs[0]))
            checks2.append(b2.check(xs[0]))
        self.assertEqual(checks1, [True, True, True, True, True])
        self.assertEqual(checks2, [True, True, True, True, True])

    def test_step_sizes(self):
        # Tests step sizes can be set and obtained
        opt = method([0], sigma0=123)
        self.assertEqual(opt.min_step_size(), 123 * 1e-3)
        self.assertIsNone(opt.max_step_size())

        opt.set_min_step_size(12)
        self.assertEqual(opt.min_step_size(), 12)
        self.assertIsNone(opt.max_step_size())

        opt.set_max_step_size(13)
        self.assertEqual(opt.min_step_size(), 12)
        self.assertEqual(opt.max_step_size(), 13)

        opt.set_min_step_size(None)
        self.assertIsNone(opt.min_step_size())
        self.assertEqual(opt.max_step_size(), 13)

        opt.set_max_step_size(None)
        self.assertIsNone(opt.min_step_size())
        self.assertIsNone(opt.max_step_size())

        # Delayed error
        opt.set_min_step_size(10)
        opt.set_max_step_size(5)
        opt.set_min_step_size(1)
        opt.set_max_step_size(0)
        opt.set_min_step_size(8)
        self.assertRaisesRegex(
            Exception, 'Max step size must be larger than min', opt.ask)

    def test_hyper_parameter_interface(self):
        # Tests the hyper parameter interface for this optimiser.
        opt = method([0])
        self.assertEqual(opt.n_hyper_parameters(), 2)
        opt.set_hyper_parameters([123, 456])
        self.assertEqual(opt.min_step_size(), 123)
        self.assertEqual(opt.max_step_size(), 456)
        opt.set_hyper_parameters([234, None])
        self.assertEqual(opt.min_step_size(), 234)
        self.assertEqual(opt.max_step_size(), None)

    def test_logging(self):

        # Test with logpdf
        r, x, s = self.problem()
        opt = pints.OptimisationController(r, x, s, method=method)
        opt.set_log_to_screen(True)
        opt.set_max_unchanged_iterations(None)
        opt.set_max_iterations(2)
        with StreamCapture() as c:
            opt.run()
        lines = c.text().splitlines()
        exp = log_1.splitlines()
        self.assertEqual(lines[0], exp[0])
        self.assertEqual(lines[1], exp[1])
        self.assertEqual(lines[2], exp[2])
        self.assertEqual(lines[3], exp[3])
        self.assertEqual(lines[4][:-3], exp[4][:-3])
        self.assertEqual(lines[5][:-3], exp[5][:-3])

        # Test with min and max steps
        r, x, s = self.problem()
        opt = pints.OptimisationController(r, x, s, method=method)
        opt.set_log_to_screen(True)
        opt.set_max_unchanged_iterations(None)
        opt.set_max_iterations(4)
        opt.set_log_interval(1)
        opt.optimiser().set_min_step_size(0.03)
        opt.optimiser().set_max_step_size(0.11)
        with StreamCapture() as c:
            opt.run()
        lines = c.text().splitlines()
        exp = log_2.splitlines()
        self.assertEqual(lines[0], exp[0])
        self.assertEqual(lines[1], exp[1])
        self.assertEqual(lines[2], exp[2])
        self.assertEqual(lines[3], exp[3])
        self.assertEqual(lines[4][:-3], exp[4][:-3])
        self.assertEqual(lines[5][:-3], exp[5][:-3])
        self.assertEqual(lines[6][:-3], exp[6][:-3])
        self.assertEqual(lines[7][:-3], exp[7][:-3])

    def test_name(self):
        # Test the name() method.
        opt = method(np.array([0]))
        self.assertEqual(opt.name(), 'iRprop-')
        self.assertTrue(opt.needs_sensitivities())


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()

