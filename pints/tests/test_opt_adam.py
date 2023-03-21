#!/usr/bin/env python3
#
# Tests the API of the Adam optimiser.
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
method = pints.Adam


log_1 = '''
Minimising error measure
Using Adam
Running in sequential mode.
Iter. Eval. Best      Current   b1        b2        Time m:s
0     1      0.02      0.02      0.9       0.999      0:00.0
1     2      5e-17     5e-17     0.81      0.998001   0:00.0
2     3      5e-17     0.00898   0.729     0.997003   0:00.0
'''.strip()


class TestAdam(unittest.TestCase):
    """
    Tests the API of the Adam optimiser.
    """
    def setUp(self):
        """ Called before every test """
        np.random.seed(1)

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
        self.assertLess(abs(found_parameters[0]), 1e-8)
        self.assertLess(abs(found_parameters[1]), 1e-8)

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
        opt = method([0])
        self.assertEqual(opt.n_hyper_parameters(), 0)

    def test_logging(self):

        # Test with logpdf
        r, x, s = self.problem()
        opt = pints.OptimisationController(r, x, s, method=method)
        opt.set_log_to_screen(True)
        opt.set_max_unchanged_iterations(None)
        opt.set_max_iterations(3)
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
        self.assertEqual(lines[6][:-3], exp[6][:-3])

    def test_name(self):
        # Test the name() method.
        opt = method(np.array([0]))
        self.assertEqual(opt.name(), 'Adam')
        self.assertTrue(opt.needs_sensitivities())


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()

