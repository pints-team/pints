#!/usr/bin/env python3
#
# Tests the pints.OptimisationController class
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import warnings

import numpy as np

import pints
import pints.toy

from shared import StreamCapture, TemporaryDirectory

debug = False
method = pints.XNES


class TestOptimisationController(unittest.TestCase):
    """
    Tests shared optimisation properties.
    """

    def setUp(self):
        """ Called before every test """
        np.random.seed(1)

    def test_best_vs_guessed(self):
        # Tests tracking and logging of best and guessed values

        # Set up a problem
        model = pints.toy.LogisticModel()
        real = model.suggested_parameters()
        times = model.suggested_times()
        values = model.simulate(real, times)
        values += np.random.normal(0, 10, values.shape)
        problem = pints.SingleOutputProblem(model, times, values)
        f = pints.SumOfSquaresError(problem)
        b = pints.RectangularBoundaries([0, 200], [1, 1000])
        x = [0, 700]

        # Check getting and setting tracking method
        np.random.seed(123)
        opt = pints.OptimisationController(
            f, x, boundaries=b, method=pints.SNES)
        self.assertFalse(opt.f_guessed_tracking())
        opt.set_f_guessed_tracking(True)
        self.assertTrue(opt.f_guessed_tracking())
        opt.set_f_guessed_tracking(False)
        self.assertFalse(opt.f_guessed_tracking())

        # Check f_best and f_guessed with callback
        fb, fg = [], []

        def cb(i, opt):
            fb.append(opt.f_best())
            fg.append(opt.f_guessed())

        # Run and check the logged values
        opt.set_callback(cb)
        opt.set_log_to_screen(False)
        opt.set_log_interval(1)
        with TemporaryDirectory() as d:
            p = d.path('out.csv')
            opt.set_log_to_file(p, csv=True)
            x1, f1 = opt.run()
            csv = np.genfromtxt(p, delimiter=',', skip_header=1)[:-1]
            lb = csv[:, 2]
            lg = csv[:, 3]
            del csv

        fb, fg = np.array(fb), np.array(fg)

        if debug:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.semilogy()
            plt.plot(fb, label='best, callback')
            plt.plot(fg, label='guessed, callback')
            plt.plot(lb, '--', label='best, logged')
            plt.plot(lg, '--', label='guessed, logged')
            plt.legend()
            plt.show()

        self.assertTrue(np.all(fb == lb))
        self.assertTrue(np.all(fg == lg))
        self.assertFalse(np.all(lb == lg))

        # Run again, but checking on f_guessed
        np.random.seed(123)
        opt2 = pints.OptimisationController(
            f, x, boundaries=b, method=pints.SNES)
        opt2.set_log_to_screen(False)
        opt2.set_f_guessed_tracking(True)
        x2, f2 = opt2.run()
        self.assertNotEqual(opt.iterations(), opt2.iterations())
        self.assertAlmostEqual(f1, f2)

    def test_callback(self):
        # Tests running with a callback method

        # Define callback that just stores the argument(s) it was called with
        args = []

        def cb(*arg):
            args.append(arg)

        # Set up a controller
        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x0 = np.array([0, 1.01])
        s = 0.01
        opt = pints.OptimisationController(r, x0, s, method=method)
        opt.set_log_to_screen(False)
        opt.set_max_unchanged_iterations(None)
        opt.set_max_iterations(10)

        # Pass in an invalid value
        self.assertRaisesRegex(
            ValueError, 'None or a callable', opt.set_callback, 3)

        # Now test using it correctly
        opt.set_callback(None)
        opt.set_callback(cb)
        opt.run()

        # Ensure callback was called at each iteration
        self.assertEqual(len(args), opt.iterations())

        # Ensure first argument was iteration count
        a = np.array([arg[0] for arg in args])
        self.assertTrue(np.all(a == np.arange(opt.iterations())))

        # Ensure second argument was always the optimisation method
        b = tuple(set([arg[1] for arg in args]))
        self.assertEqual(len(b), 1)
        self.assertIs(b[0], opt.optimiser())

        # Check unsetting works
        args.clear()
        self.assertEqual(len(args), 0)
        opt = pints.OptimisationController(r, x0, s, method=method)
        opt.set_log_to_screen(False)
        opt.set_max_unchanged_iterations(None)
        opt.set_max_iterations(10)
        opt.set_callback(cb)
        opt.set_callback(None)
        opt.run()
        self.assertEqual(len(args), 0)

    def test_optimise(self):
        # Tests :meth: `pints.optimise()`.

        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x = np.array([0, 1.01])
        s = 0.01
        b = pints.RectangularBoundaries([-0.01, 0.95], [0.01, 1.05])
        with StreamCapture():
            x, f = pints.optimise(r, x, s, b, method=pints.XNES)
        self.assertEqual(x.shape, (2, ))
        self.assertTrue(f < 1e-6)

    def test_transform(self):
        # Test optimisation with parameter transformation.

        # Test with LogPDF
        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x0 = np.array([0, 1.01])
        b = pints.RectangularBoundaries([-0.01, 0.95], [0.01, 1.05])
        s = 0.01
        t = pints.RectangularBoundariesTransformation(b)
        with warnings.catch_warnings(record=True):
            opt = pints.OptimisationController(r, x0, s, b, t, method)
        opt.set_log_to_screen(False)
        opt.set_max_unchanged_iterations(None)
        opt.set_max_iterations(10)
        opt.run()

        # Test with ErrorMeasure
        r = pints.toy.ParabolicError()
        x0 = [0.1, 0.1]
        b = pints.RectangularBoundaries([-1, -1], [1, 1])
        s = 0.1
        t = pints.RectangularBoundariesTransformation(b)
        with warnings.catch_warnings(record=True):
            opt = pints.OptimisationController(r, x0, s, b, t, method)
        opt.set_log_to_screen(False)
        opt.set_max_unchanged_iterations(None)
        opt.set_max_iterations(10)
        x, _ = opt.run()

        # Test output is detransformed
        self.assertEqual(x.shape, (2, ))
        self.assertTrue(b.check(x))

    def test_stopping_max_evaluations(self):
        # Runs an optimisation with the max_fevals stopping criterion.

        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x = np.array([0, 1.01])
        b = pints.RectangularBoundaries([-0.01, 0.95], [0.01, 1.05])
        s = 0.01
        opt = pints.OptimisationController(r, x, s, b, method=method)
        opt.set_log_to_screen(True)
        opt.set_max_unchanged_iterations(None)
        opt.set_max_evaluations(10)
        self.assertEqual(opt.max_evaluations(), 10)
        self.assertRaises(ValueError, opt.set_max_evaluations, -1)
        with StreamCapture() as c:
            opt.run()
            self.assertIn('Halting: Maximum number of evaluations', c.text())

    def test_stopping_max_iterations(self):
        # Runs an optimisation with the max_iter stopping criterion.

        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x = np.array([0, 1.01])
        b = pints.RectangularBoundaries([-0.01, 0.95], [0.01, 1.05])
        s = 0.01
        opt = pints.OptimisationController(r, x, s, b, method=method)
        opt.set_log_to_screen(True)
        opt.set_max_unchanged_iterations(None)
        opt.set_max_iterations(10)
        self.assertEqual(opt.max_iterations(), 10)
        self.assertRaises(ValueError, opt.set_max_iterations, -1)
        with StreamCapture() as c:
            opt.run()
            self.assertIn('Halting: Maximum number of iterations', c.text())

    def test_logging(self):

        # Test with logpdf
        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x = np.array([0, 1.01])
        b = pints.RectangularBoundaries([-0.01, 0.95], [0.01, 1.05])
        s = 0.01
        opt = pints.OptimisationController(r, x, s, b, method=method)
        opt.set_log_to_screen(True)
        opt.set_max_unchanged_iterations(None)
        opt.set_log_interval(3)
        opt.set_max_iterations(10)
        with StreamCapture() as c:
            opt.run()
        lines = c.text().splitlines()
        self.assertEqual(lines[0], 'Maximising LogPDF')
        self.assertEqual(
            lines[1], 'Using Exponential Natural Evolution Strategy (xNES)')
        self.assertEqual(lines[2], 'Running in sequential mode.')
        self.assertEqual(lines[3], 'Population size: 6')
        self.assertEqual(lines[4],
                         'Iter. Eval. Best      Current   Time m:s')
        self.assertEqual(lines[5][:-3],
                         '0     3     -4.140462 -4.140462   0:0')
        self.assertEqual(lines[6][:-3],
                         '1     6     -4.140462 -4.140465   0:0')
        self.assertEqual(lines[7][:-3],
                         '2     11    -4.140462 -4.140462   0:0')
        self.assertEqual(lines[8][:-3],
                         '3     16    -4.140462 -4.140466   0:0')
        self.assertEqual(lines[9][:-3],
                         '6     33    -4.140462 -4.140462   0:0')
        self.assertEqual(lines[10][:-3],
                         '9     51    -4.140462 -4.140462   0:0')
        self.assertEqual(lines[11][:-3],
                         '10    51    -4.140462 -4.140462   0:0')
        self.assertEqual(
            lines[12], 'Halting: Maximum number of iterations (10) reached.')

        # Invalid log interval
        self.assertRaises(ValueError, opt.set_log_interval, 0)

        # Test with error measure
        r = pints.toy.RosenbrockError()
        x = np.array([1.01, 1.01])
        opt = pints.OptimisationController(r, x, method=pints.SNES)
        opt.set_log_to_screen(True)
        opt.set_max_unchanged_iterations(None)
        opt.set_log_interval(4)
        opt.set_max_iterations(11)
        opt.optimiser().set_population_size(4)
        with StreamCapture() as c:
            opt.run()
        lines = c.text().splitlines()
        self.assertEqual(lines[0], 'Minimising error measure')
        self.assertEqual(
            lines[1], 'Using Seperable Natural Evolution Strategy (SNES)')
        self.assertEqual(lines[2], 'Running in sequential mode.')
        self.assertEqual(lines[3], 'Population size: 4')
        self.assertEqual(lines[4],
                         'Iter. Eval. Best      Current   Time m:s')
        self.assertEqual(lines[5][:-3],
                         '0     4      6.471867  6.471867   0:0')
        self.assertEqual(lines[6][:-3],
                         '1     8      6.471867  14.54646   0:0')
        self.assertEqual(lines[7][:-3],
                         '2     12     0.0949    0.0949     0:0')
        self.assertEqual(lines[8][:-3],
                         '3     16     0.0949    0.288      0:0')
        self.assertEqual(lines[9][:-3],
                         '4     20     0.0949    0.11       0:0')
        self.assertEqual(lines[10][:-3],
                         '8     36     0.0165    5.504433   0:0')
        self.assertEqual(lines[11][:-3],
                         '11    44     0.0165    3.601763   0:0')
        self.assertEqual(
            lines[12], 'Halting: Maximum number of iterations (11) reached.')

        # Invalid log interval
        self.assertRaises(ValueError, opt.set_log_interval, 0)

    def test_stopping_max_unchanged(self):
        # Runs an optimisation with the max_unchanged stopping criterion.
        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x = np.array([0, 1.01])
        b = pints.RectangularBoundaries([-0.01, 0.95], [0.01, 1.05])
        s = 0.01
        opt = pints.OptimisationController(r, x, s, b, method=method)
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
        # Runs an optimisation with the threshold stopping criterion.

        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x = np.array([0.008, 1.01])
        b = pints.RectangularBoundaries([-0.01, 0.95], [0.01, 1.05])
        s = 0.01
        opt = pints.OptimisationController(r, x, s, b, method=method)
        opt.set_log_to_screen(True)
        opt.set_max_iterations(None)
        opt.set_max_unchanged_iterations(None)
        opt.set_threshold(5)
        self.assertEqual(opt.threshold(), 5)
        with StreamCapture() as c:
            opt.run()
            self.assertIn(
                'Halting: Objective function crossed threshold', c.text())

    def test_stopping_no_criterion(self):
        # Tries to run an optimisation with the no stopping criterion.

        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x = np.array([0, 1.01])
        b = pints.RectangularBoundaries([-0.01, 0.95], [0.01, 1.05])
        s = 0.01
        opt = pints.OptimisationController(r, x, s, b, method=method)
        opt.set_log_to_screen(debug)
        opt.set_max_iterations(None)
        opt.set_max_unchanged_iterations(None)
        self.assertRaises(ValueError, opt.run)

    def test_set_population_size(self):
        # Tests the set_population_size method for this optimiser.

        r = pints.toy.RosenbrockError()
        x = np.array([1.01, 1.01])
        opt = pints.OptimisationController(r, x, method=method)
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
        # Test parallelised running.

        r = pints.toy.RosenbrockError()
        x = np.array([1.1, 1.1])
        b = pints.RectangularBoundaries([0.5, 0.5], [1.5, 1.5])

        # Run with guessed number of cores
        opt = pints.OptimisationController(r, x, boundaries=b, method=method)
        opt.set_max_iterations(10)
        opt.set_log_to_screen(debug)
        opt.set_parallel(False)
        self.assertIs(opt.parallel(), False)
        opt.set_parallel(True)
        self.assertTrue(type(opt.parallel()) == int)
        self.assertTrue(opt.parallel() >= 1)
        opt.run()

        # Run with explicit number of cores
        opt = pints.OptimisationController(r, x, boundaries=b, method=method)
        opt.set_max_iterations(10)
        opt.set_log_to_screen(debug)
        opt.set_parallel(4)
        self.assertTrue(type(opt.parallel()) == int)
        self.assertEqual(opt.parallel(), 4)
        opt.run()

    def test_deprecated_alias(self):
        # Tests Optimisation()
        r = pints.toy.RosenbrockError()
        x = np.array([1.1, 1.1])
        b = pints.RectangularBoundaries([0.5, 0.5], [1.5, 1.5])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            opt = pints.Optimisation(r, x, boundaries=b, method=method)
        self.assertEqual(len(w), 1)
        self.assertIn('deprecated', str(w[-1].message))
        self.assertIsInstance(opt, pints.OptimisationController)

    def test_post_run_statistics(self):

        # Test the methods to return statistics, post-run.
        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x = np.array([0, 1.01])
        b = pints.RectangularBoundaries([-0.01, 0.95], [0.01, 1.05])
        s = 0.01
        opt = pints.OptimisationController(r, x, s, b, method=method)
        opt.set_log_to_screen(False)
        opt.set_max_unchanged_iterations(50, 1e-11)

        # Before run methods return None
        self.assertIsNone(opt.iterations())
        self.assertIsNone(opt.evaluations())
        self.assertIsNone(opt.time())

        t = pints.Timer()
        opt.run()
        t_upper = t.time()

        self.assertEqual(opt.iterations(), 84)
        self.assertEqual(opt.evaluations(), 495)

        # Time after run is greater than zero
        self.assertIsInstance(opt.time(), float)
        self.assertGreater(opt.time(), 0)
        self.assertGreater(t_upper, opt.time())

    def test_exception_on_multi_use(self):
        # Controller should raise an exception if use multiple times

        r = pints.toy.TwistedGaussianLogPDF(2, 0.01)
        x = np.array([0, 1.01])
        b = pints.RectangularBoundaries([-0.01, 0.95], [0.01, 1.05])
        s = 0.01
        opt = pints.OptimisationController(r, x, s, b, method=method)
        opt.set_log_to_screen(False)
        opt.set_max_unchanged_iterations(None)
        opt.set_max_iterations(10)
        opt.run()
        self.assertRaisesRegex(
            RuntimeError, 'Controller is valid for single use only', opt.run)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
