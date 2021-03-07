#!/usr/bin/env python3
#
# Tests ellipsoidal nested sampler.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np

import pints
import pints.toy
from pints._nested._multinest import EllipsoidTree
from pints._nested.__init__ import Ellipsoid

# Unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


# class TestMultiNestSampler(unittest.TestCase):
#     """
#     Unit (not functional!) tests for :class:`MultinestSampler`.
#     """
#
#     @classmethod
#     def setUpClass(cls):
#         """ Prepare for the test. """
#         # Create toy model
#         model = pints.toy.LogisticModel()
#         cls.real_parameters = [0.015, 500]
#         times = np.linspace(0, 1000, 1000)
#         values = model.simulate(cls.real_parameters, times)
#
#         # Add noise
#         np.random.seed(1)
#         cls.noise = 10
#         values += np.random.normal(0, cls.noise, values.shape)
#         cls.real_parameters.append(cls.noise)
#
#         # Create an object with links to the model and time series
#         problem = pints.SingleOutputProblem(model, times, values)
#
#         # Create a uniform prior over both the parameters and the new noise
#         # variable
#         cls.log_prior = pints.UniformLogPrior(
#             [0.01, 400],
#             [0.02, 600]
#         )
#
#         # Create a log-likelihood
#         cls.log_likelihood = pints.GaussianKnownSigmaLogLikelihood(
#             problem, cls.noise)
#
#     def test_getters_and_setters(self):
#         # tests various get() and set() methods.
#         controller = pints.NestedController(self.log_likelihood,
#                                             self.log_prior,
#                                             method=pints.MultinestSampler)
#         self.assertEqual(controller.sampler().f_s_threshold(), 1.1)
#         controller.sampler().set_f_s_threshold(4)
#         self.assertEqual(controller.sampler().f_s_threshold(), 4)
#         self.assertRaises(ValueError, controller.sampler().set_f_s_threshold,
#                           0.5)
#
#     def test_runs(self):
#         # tests that sampler runs
#         sampler = pints.NestedController(self.log_likelihood, self.log_prior,
#                                          method=pints.MultinestSampler)
#         sampler.set_iterations(100)
#         sampler.set_log_to_screen(False)
#         sampler.run()


class TestEllipsoidTree(unittest.TestCase):
    """
    Unit tests for the Ellipsoid binary tree class.
    """

    @classmethod
    def setUpClass(cls):
        # prepare for tests
        n = 1000
        gaussian = pints.MultivariateGaussianLogPrior([0, 0], [[1, 0], [0, 1]])
        cls.gaussian = gaussian
        draws = gaussian.sample(n)
        cls.draws = [gaussian.convert_to_unit_cube(x) for x in draws]

    def unit_draws(self, n):
        # helper function to produce unit draws
        draws = self.gaussian.sample(n)
        return [self.gaussian.convert_to_unit_cube(x) for x in draws]

    def test_construction(self):
        # tests construction

        # above threshold draws
        draws = self.unit_draws(20)
        EllipsoidTree(draws, 1)
        # below threshold draws
        draws = self.unit_draws(5)
        EllipsoidTree(draws, 1)

        # check errors
        # zero length points?
        self.assertRaises(ValueError, EllipsoidTree, [], 1)
        # negative iteration?
        self.assertRaises(ValueError, EllipsoidTree, self.draws, -1)
        # points within unit cube?
        gaussian = pints.MultivariateGaussianLogPrior([5, 5], [[1, 0], [0, 1]])
        draws = gaussian.sample(20)
        self.assertRaises(ValueError, EllipsoidTree, draws, 1)

    def test_getters(self):
        # tests get()

        tree = EllipsoidTree(self.draws, 100)

        # leaf nodes
        self.assertTrue(tree.n_leaf_ellipsoids() >= 1)
        leaves = tree.leaf_ellipsoids()
        self.assertEqual(tree.n_leaf_ellipsoids(), len(leaves))
        [self.assertTrue(isinstance(x, Ellipsoid)) for x in leaves]

        # bounding ellipsoid
        ellipsoid = tree.ellipsoid()
        self.assertTrue(isinstance(ellipsoid, Ellipsoid))

    def test_calculations(self):
        # tests the calculations done

        # vs
        n = 20
        iteration = 2
        draws = self.unit_draws(n)
        ellipsoidTree = EllipsoidTree(draws, iteration)
        self.assertEqual(ellipsoidTree.vs(), np.exp(-iteration / n))

        # vsk
        n1 = 5
        draws = self.unit_draws(n1)
        ellipsoid = Ellipsoid.minimum_volume_ellipsoid(draws)
        V_S = ellipsoidTree._V_S
        self.assertEqual(ellipsoidTree.vsk(ellipsoid), n1 * V_S / n)

        # more points in ellipsoid than tree?
        n1 = 21
        draws = self.unit_draws(n1)
        ellipsoid = Ellipsoid.minimum_volume_ellipsoid(draws)
        self.assertRaises(ValueError, ellipsoidTree.vsk, ellipsoid)




if __name__ == '__main__':
    unittest.main()
