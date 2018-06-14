#!/usr/bin/env python
#
# Tests the basic methods of the SNES optimiser.
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
method = pints.SNES


class TestSNES(unittest.TestCase):
    """
    Tests the basic methods of the SNES optimiser.
    """
    def __init__(self, name):
        super(TestSNES, self).__init__(name)

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
        self.cutoff = 1e-9

        # Maximum tries before it counts as failed
        self.max_tries = 3

    def test_unbounded(self):

        opt = pints.Optimisation(self.score, self.x0, method=method)
        opt.set_log_to_screen(debug)
        for i in range(self.max_tries):
            found_parameters, found_solution = opt.run()
            if found_solution < self.cutoff:
                break
        self.assertTrue(found_solution < self.cutoff)

    def test_bounded(self):

        opt = pints.Optimisation(self.score, self.x0,
                                 boundaries=self.boundaries, method=method)
        opt.set_log_to_screen(debug)
        for i in range(self.max_tries):
            found_parameters, found_solution = opt.run()
            if found_solution < self.cutoff:
                break
        self.assertTrue(found_solution < self.cutoff)

    def test_bounded_and_sigma(self):

        opt = pints.Optimisation(self.score, self.x0, self.sigma0,
                                 self.boundaries, method)
        opt.set_log_to_screen(debug)
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
        with StreamCapture() as c:
            opt.run()
            self.assertIn('Halting: No significant change', c.text())

    def test_stopping_threshold(self):

        opt = pints.Optimisation(self.score, self.x0, self.sigma0,
                                 self.boundaries, method)
        opt.set_log_to_screen(True)
        opt.set_max_iterations(None)
        opt.set_max_unchanged_iterations(None)
        opt.set_threshold(1e4 * self.cutoff)
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
        self.assertRaises(ValueError, opt.run)

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
