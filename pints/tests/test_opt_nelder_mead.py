#!/usr/bin/env python3
#
# Tests the basic methods of the Nelder-Mead optimiser.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
import unittest
import sys

import pints
import pints.toy

from shared import StreamCapture

debug = False
method = pints.NelderMead


class TestNelderMead(unittest.TestCase):
    """
    Tests the basic methods of the Nelder-Mead optimiser.
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
        opt.set_threshold(1e-3)
        opt.set_log_to_screen(debug)
        found_parameters, found_solution = opt.run()
        self.assertTrue(found_solution < 1e-3)

    @unittest.skipIf(sys.hexversion < 0x03040000, 'Python < 3.4')
    def test_bounded_warning(self):
        # Boundaries are not supported
        r, x, s, b = self.problem()

        # Rectangular boundaries
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            pints.OptimisationController(r, x, boundaries=b, method=method)
            self.assertEqual(len(w), 1)
            self.assertIn('does not support boundaries', str(w[-1].message))

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

        # Not running
        self.assertFalse(opt.running())

        # Running
        xs = opt.ask()
        fxs = [r(x) for x in xs]
        self.assertTrue(opt.running())
        opt.tell(fxs)
        self.assertTrue(opt.running())
        xs = opt.ask()
        fxs = [r(x) for x in xs]
        opt.tell(fxs)

        # Tell before ask
        self.assertRaisesRegex(
            Exception, r'ask\(\) not called before tell\(\)', opt.tell, 5)

        # Ask called twice
        opt.ask()
        self.assertRaisesRegex(
            Exception, r'ask\(\) called twice', opt.ask)

    def test_hyper_parameter_interface(self):
        # Tests the hyper parameter interface for this optimiser.
        r, x, s, b = self.problem()
        opt = pints.OptimisationController(r, x, method=method)
        m = opt.optimiser()
        self.assertEqual(m.n_hyper_parameters(), 0)

    def test_name(self):
        # Tests the name() method.
        opt = method(np.array([0, 1.01]))
        self.assertIn('Nelder-Mead', opt.name())

    def test_zeros_in_x(self):
        # Tests if the method copes with zeros in x0 (which can go wrong
        # depending on the initialisation method).

        r = pints.toy.RosenbrockError()
        x0 = [0, 0]
        opt = pints.OptimisationController(r, x0, method=method)
        opt.set_log_to_screen(False)
        x, f = opt.run()
        self.assertTrue(np.all(x == np.array([1, 1])))
        self.assertEqual(f, 0)

    def test_bad_tell(self):
        # Tests errors if wrong sizes are passed to tell

        r = pints.toy.RosenbrockError()
        e = pints.SequentialEvaluator(r)
        x0 = [0, 0]

        # Give wrong initial number
        opt = method(x0)
        xs = opt.ask()
        fxs = e.evaluate(xs)
        self.assertRaisesRegex(
            ValueError, r'of length \(1 \+ n_parameters\)', opt.tell, fxs[:-1])

        # Give wrong intermediate answer
        opt = method(x0)
        opt.tell(e.evaluate(opt.ask()))
        x = opt.ask()[0]
        fx = e.evaluate([x])
        self.assertRaisesRegex(
            ValueError, 'only a single evaluation', opt.tell, [fx, fx])

        # Give wrong answer in shrink step
        with self.assertRaisesRegex(ValueError, 'length n_parameters'):
            opt = method(x0)
            for i in range(500):
                opt.tell(e.evaluate(opt.ask()))
                if opt._shrink:
                    xs = opt.ask()
                    fxs = e.evaluate(xs)
                    opt.tell(fxs[:-1])
                    break

    def test_rosenbrock(self):
        # Tests the actions of the optimiser against a stored result

        r = pints.toy.RosenbrockError()
        x0 = [-0.75, 3.5]

        opt = pints.OptimisationController(r, x0, method=method)
        opt.set_log_to_screen(True)

        with StreamCapture() as c:
            x, f = opt.run()
            log = c.text()

        self.assertTrue(np.all(x == np.array([1, 1])))
        self.assertEqual(f, 0)

        exp_lines = (
            'Minimising error measure',
            'Using Nelder-Mead',
            'Running in sequential mode.',
            'Iter. Eval. Best      Current   Time m:s',
            '0     3      865.9531  865.9531   0:00.0',
            '1     4      832.5452  832.5452   0:00.0',
            '2     5      832.5452  832.5452   0:00.0',
            '3     6      628.243   628.243    0:00.0',
            '20    23     4.95828   4.95828    0:00.0',
            '40    43     3.525867  3.525867   0:00.0',
            '60    63     2.377579  2.377579   0:00.0',
            '80    83     1.114115  1.114115   0:00.0',
            '100   103    0.551     0.551      0:00.0',
            '120   123    0.237     0.237      0:00.0',
            '140   143    0.0666    0.0666     0:00.0',
            '160   163    0.00181   0.00181    0:00.0',
            '180   183    6.96e-06  6.96e-06   0:00.0',
            '200   203    2.66e-08  2.66e-08   0:00.0',
            '220   223    5.06e-11  5.06e-11   0:00.0',
            '240   243    2.43e-15  2.43e-15   0:00.0',
            '260   263    5.58e-18  5.58e-18   0:00.0',
            '280   283    7.74e-20  7.74e-20   0:00.0',
            '300   303    6.66e-23  6.66e-23   0:00.0',
            '320   323    1.86e-25  1.86e-25   0:00.0',
            '340   343    3.16e-28  3.16e-28   0:00.0',
            '360   364    3.08e-31  3.08e-31   0:00.0',
            '380   390    0         0          0:00.0',
            '400   416    0         0          0:00.0',
            '420   443    0         0          0:00.0',
            '428   452    0         0          0:00.0',
            'Halting: No significant change for 200 iterations.',
        )

        # Compare lenght of log
        log_lines = [line.rstrip() for line in log.splitlines()]
        self.assertEqual(len(log_lines), len(exp_lines))

        # Compare log lines, ignoring time bit (unles it's way too slow)
        for line1, line2 in zip(log_lines, exp_lines):
            if line2[-6:] == '0:00.0':
                line1 = line1[:-6]
                line2 = line2[:-6]
            self.assertEqual(line1, line2)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
