#!/usr/bin/env python3
#
# Tests the basic methods of the CMAES optimiser.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import pints.io
import pints.toy
import unittest
import numpy as np

debug = False
method = pints.CMAES


class TestCMAES(unittest.TestCase):
    """
    Tests the basic methods of the CMAES optimiser.
    """
    def __init__(self, name):
        super(TestCMAES, self).__init__(name)

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
        with pints.io.StreamCapture() as c:
            opt.run()
            self.assertIn('Halting: Maximum number of iterations', c.text())

    def test_stopping_max_unchanged(self):

        opt = pints.Optimisation(self.score, self.x0, self.sigma0,
                                 self.boundaries, method)
        opt.set_log_to_screen(True)
        opt.set_max_iterations(None)
        opt.set_max_unchanged_iterations(2)
        with pints.io.StreamCapture() as c:
            opt.run()
            self.assertIn('Halting: No significant change', c.text())

    def test_stopping_threshold(self):

        opt = pints.Optimisation(self.score, self.x0, self.sigma0,
                                 self.boundaries, method)
        opt.set_log_to_screen(True)
        opt.set_max_iterations(None)
        opt.set_max_unchanged_iterations(None)
        opt.set_threshold(1e4 * self.cutoff)
        with pints.io.StreamCapture() as c:
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

    def test_stopping_on_ill_conditioned_covariance_matrix(self):
        from scipy.integrate import odeint

        def OnePopControlODE(y, t, p):
            a, b, c = p
            dydt = np.zeros(y.shape)
            k = (a - b) / c * (y[0] + y[1])
            dydt[0] = a * y[0] - b * y[0] - k * y[0]
            dydt[1] = k * y[0] - b * y[1]
            return dydt

        class Model(pints.ForwardModel):

            def simulate(self, parameters, times, n_derivatives=0):
                y0 = [2000000, 0]
                solution = odeint(
                    OnePopControlODE, y0, times, args=(parameters,))
                return np.sum(np.array(solution), axis=1)

            def n_parameters(self):
                return 3

        model = Model()
        times = [0, 0.5, 2, 4, 8, 24]
        values = [2e6, 3.9e6, 3.1e7, 3.7e8, 1.6e9, 1.6e9]
        problem = pints.SingleOutputProblem(model, times, values)
        score = pints.SumOfSquaresError(problem)
        x0 = [2.5, 0.0001, 5e6]
        with pints.io.StreamCapture() as c:
            pints.optimise(score, x0)
        self.assertTrue('Ill-conditioned covariance matrix' in c.text())


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
