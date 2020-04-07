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
from shared import StreamCapture

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
        r, x0, sigma0, b = self.problem()
        
        # Show animation of progress
        if False:
            x0 = [0.7, 0.6]
            
            import matplotlib.pyplot as plt
            from matplotlib.patches import Ellipse
            
            plt.figure(figsize=(10, 10))
            ax = plt.subplot(1, 1, 1)
            plt.xlim(-0.6, 1)
            plt.ylim(-0.6, 1)
            plt.plot([0], [0], 'o', label='true')
            
            x = np.arange(-1.5, 1.5, 0.1)
            X, Y = np.meshgrid(x, x)
            f = np.copy(X)
            for i, xx in enumerate(x):
                for j, yy in enumerate(x):
                    f[i, j] = np.sqrt(r([xx, yy]))
            plt.contour(X, Y, f)
            
            opt = pints.BareCMAES(x0, sigma0)
            
            e = pints.ParallelEvaluator(r)

            x1, y1 = x0
            for i in range(60):
                
                # Plot jump in mean of distribution
                x2, y2 = opt.mean()
                plt.plot([x1, x2], [y1, y2], 'x-', color='tab:blue')
                x1, y1 = x2, y2
                
                # Draw ellipse of covariance matrix
                # width = 2 * first eigenvalue
                # height = 2 * second eigenvalue
                # rotation = first eigenvector
                R, S = opt.cov(True)            
                w = 2 * S[0, 0]     # First eigenvalue
                h = 2 * S[1, 1]     # Second eigenvalue
                t = np.arctan2(R[1, 0], R[0, 0])  # Angle of first eigenvector
                ellipse = Ellipse((x2, y2), w, h, t * 180 / np.pi, **{
                    'facecolor': 'none',
                    'edgecolor': 'red',
                    'alpha': 1,
                })
                ax.add_patch(ellipse)

                # Move to next point            
                opt.tell(e.evaluate(opt.ask()))
                
                plt.pause(0.2)
                ellipse.remove()
                
            return

        
        # Temporary test        
        x0 = [0.7, 0.6]
        opt = pints.BareCMAES(x0, sigma0)
        e = pints.ParallelEvaluator(r)
        for i in range(10):
            opt.tell(e.evaluate(opt.ask()))
        a, b = opt.mean()
        print(pints.strfloat(a))
        print(pints.strfloat(b))
        assert a == -4.93386229572480295e-03
        assert b == 6.43128695469029760e-02


 

        
        '''
        r, x, s, b = self.problem()
        opt = pints.OptimisationController(r, x, method=method)
        opt.set_threshold(1e-3)
        opt.set_log_to_screen(debug)
        found_parameters, found_solution = opt.run()
        self.assertTrue(found_solution < 1e-3)
        '''

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
    '''
    def test_bounded_and_sigma(self):
        # Runs an optimisation without boundaries and sigma.
        r, x, s, b = self.problem()
        opt = pints.OptimisationController(r, x, s, b, method)
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
'''
    def test_name(self):
        # Test the name() method.
        opt = method(np.array([0, 1.01]))
        self.assertIn('Bare-bones CMA-ES', opt.name())


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
        import logging
        logging.basicConfig(level=logging.DEBUG)
    unittest.main()
