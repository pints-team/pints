#!/usr/bin/env python3
#
# Tests the basic methods of the CMAES optimiser.
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
from shared import StreamCapture

debug = False
method = pints.CMAES


class TestCMAES(unittest.TestCase):
    """
    Tests the basic methods of the CMAES optimiser.
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

    def test_bounded(self):
        # Runs an optimisation with boundaries.
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
        # Runs an optimisation without boundaries and sigma.
        r, x, s, b = self.problem()
        opt = pints.OptimisationController(r, x, s, b, method=method)
        opt.set_threshold(1e-3)
        opt.set_log_to_screen(debug)
        found_parameters, found_solution = opt.run()
        self.assertTrue(found_solution < 1e-3)

    @unittest.skip('Newer versions of cma no longer trigger this condition')
    def test_stopping_on_ill_conditioned_covariance_matrix(self):
        # Tests that ill conditioned covariance matrices are detected.
        from scipy.integrate import odeint
        #TODO: A quicker test-case for this would be great!

        def OnePopControlODE(y, t, p):
            a, b, c = p
            dydt = np.zeros(y.shape)
            k = (a - b) / c * (y[0] + y[1])
            dydt[0] = a * y[0] - b * y[0] - k * y[0]
            dydt[1] = k * y[0] - b * y[1]
            return dydt

        class Model(pints.ForwardModel):

            def simulate(self, parameters, times):
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
        x = [3.42, -0.21, 5e6]
        opt = pints.OptimisationController(score, x, method=method)
        with StreamCapture() as c:
            opt.run()
        self.assertTrue('Ill-conditioned covariance matrix' in c.text())

    def test_ask_tell(self):
        # Tests ask-and-tell related error handling.
        r, x, s, b = self.problem()
        opt = method(x, boundaries=b)

        # Stop called when not running
        self.assertFalse(opt.stop())

        # Best position and score called before run
        self.assertEqual(list(opt.x_best()), list(x))
        self.assertEqual(list(opt.x_guessed()), list(x))
        self.assertEqual(opt.f_best(), np.inf)
        self.assertEqual(opt.f_guessed(), np.inf)

        # Test deprecated xbest and fbest
        self.assertEqual(list(opt.xbest()), list(x))
        self.assertEqual(opt.fbest(), np.inf)

        # Tell before ask
        self.assertRaisesRegex(
            Exception, r'ask\(\) not called before tell\(\)', opt.tell, 5)

    def test_is_default(self):
        # Checks this is the default optimiser.
        r, x, s, b = self.problem()
        opt = pints.OptimisationController(r, x)
        self.assertIsInstance(opt.optimiser(), method)

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
        self.assertIn('CMA-ES', opt.name())


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()

