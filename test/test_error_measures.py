#!/usr/bin/env python3
#
# Tests the error measure classes.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import division
import pints
import pints.toy
import unittest
import numpy as np


class MiniProblem(pints.SingleSeriesProblem):
    def __init__(self):
        self._t = pints.vector([1, 2, 3])
        self._v = pints.vector([-1, 2, 3])

    def dimension(self):
        return 3

    def evaluate(self, parameters):
        return np.array(parameters)

    def times(self):
        return self._t

    def values(self):
        return self._v


class MultiMiniProblem(pints.MultiSeriesProblem):
    def __init__(self):
        self._t = pints.vector([1, 2, 3])
        self._v = pints.matrix2d(
            np.array([[-1, 2, 3], [-1, 2, 3]]).swapaxes(0, 1))

    def dimension(self):
        return 3

    def n_outputs(self):
        return 2

    def evaluate(self, parameters):
        return np.array([parameters, parameters]).swapaxes(0, 1)

    def times(self):
        return self._t

    def values(self):
        return self._v


class BigMiniProblem(MiniProblem):
    def __init__(self):
        super(BigMiniProblem, self).__init__()
        self._t = pints.vector([1, 2, 3, 4, 5, 6])
        self._v = pints.vector([-1, 2, 3, 4, 5, -6])

    def dimension(self):
        return 6


class BadMiniProblem(MiniProblem):
    def __init__(self, bad_value=float('inf')):
        super(BadMiniProblem, self).__init__()
        self._v = pints.vector([bad_value, 2, -3])

    def dimension(self):
        return 3


class BadErrorMeasure(pints.ErrorMeasure):
    def __init__(self, bad_value=float('-inf')):
        super(BadErrorMeasure, self).__init__()
        self._v = bad_value

    def dimension(self):
        return 3

    def __call__(self, parameters):
        return self._v


class MiniLogPDF(pints.LogPDF):
    def dimension(self):
        return 3

    def __call__(self, parameters):
        return 10


class TestErrorMeasures(unittest.TestCase):
    """
    Tests the ErrorMeasure classes
    """
    def __init__(self, name):
        super(TestErrorMeasures, self).__init__(name)

    def test_mean_squared_error(self):
        p = MiniProblem()
        e = pints.MeanSquaredError(p)
        self.assertEqual(e.dimension(), 3)
        float(e([1, 2, 3]))
        self.assertEqual(e([-1, 2, 3]), 0)
        self.assertNotEqual(np.all(e([1, 2, 3])), 0)
        x = [0, 0, 0]
        y = (1 + 4 + 9) / 3
        self.assertAlmostEqual(e(x), y)
        x = [1, 1, 1]
        y = (4 + 1 + 4) / 3
        self.assertEqual(e(x), y)

        p = MultiMiniProblem()
        e = pints.MeanSquaredError(p)
        self.assertEqual(e.dimension(), 3)
        float(e([1, 2, 3]))
        self.assertEqual(e([-1, 2, 3]), 0)
        self.assertNotEqual(np.all(e([1, 2, 3])), 0)
        x = [0, 0, 0]
        y = (1 + 4 + 9) / 3
        self.assertAlmostEqual(e(x), y)
        x = [1, 1, 1]
        y = (4 + 1 + 4) / 3
        self.assertEqual(e(x), y)

    def test_probability_based_error(self):
        p = MiniLogPDF()
        e = pints.ProbabilityBasedError(p)
        self.assertEqual(e.dimension(), 3)
        self.assertEqual(e([1, 2, 3]), -10)
        p = MiniProblem()
        self.assertRaises(ValueError, pints.ProbabilityBasedError, p)

    def test_root_mean_squared_error(self):
        p = MiniProblem()
        e = pints.RootMeanSquaredError(p)
        self.assertEqual(e.dimension(), 3)
        float(e([1, 2, 3]))
        self.assertEqual(e([-1, 2, 3]), 0)
        self.assertNotEqual(np.all(e([1, 2, 3])), 0)
        x = [0, 0, 0]
        y = np.sqrt((1 + 4 + 9) / 3)
        self.assertAlmostEqual(e(x), y)
        x = [1, 1, 1]
        y = np.sqrt((4 + 1 + 4) / 3)
        self.assertEqual(e(x), y)

        p = MultiMiniProblem()
        self.assertRaises(ValueError, pints.RootMeanSquaredError, p)

    def test_sum_of_squares_error(self):
        p = MiniProblem()
        e = pints.SumOfSquaresError(p)
        self.assertEqual(e.dimension(), 3)
        float(e([1, 2, 3]))
        self.assertEqual(e([-1, 2, 3]), 0)
        self.assertNotEqual(np.all(e([1, 2, 3])), 0)
        x = [0, 0, 0]
        y = 1 + 4 + 9
        self.assertEqual(e(x), y)
        x = [1, 1, 1]
        y = 4 + 1 + 4
        self.assertEqual(e(x), y)

        p = MultiMiniProblem()
        e = pints.SumOfSquaresError(p)
        self.assertEqual(e.dimension(), 3)
        float(e([1, 2, 3]))
        self.assertEqual(e([-1, 2, 3]), 0)
        self.assertNotEqual(np.all(e([1, 2, 3])), 0)
        x = [0, 0, 0]
        y = 1 + 4 + 9
        self.assertEqual(e(x), 2 * y)
        x = [1, 1, 1]
        y = 4 + 1 + 4
        self.assertEqual(e(x), 2 * y)

    def test_sum_of_errors(self):
        e1 = pints.SumOfSquaresError(MiniProblem())
        e2 = pints.MeanSquaredError(MiniProblem())
        e3 = pints.RootMeanSquaredError(BigMiniProblem())
        e4 = pints.SumOfSquaresError(BadMiniProblem())

        # Basic use
        e = pints.SumOfErrors([e1, e2])
        x = [0, 0, 0]
        self.assertEqual(e.dimension(), 3)
        self.assertEqual(e(x), e1(x) + e2(x))
        e = pints.SumOfErrors([e1, e2], [3.1, 4.5])
        x = [0, 0, 0]
        self.assertEqual(e.dimension(), 3)
        self.assertEqual(e(x), 3.1 * e1(x) + 4.5 * e2(x))
        e = pints.SumOfErrors(
            [e1, e1, e1, e1, e1, e1], [1, 2, 3, 4, 5, 6])
        self.assertEqual(e.dimension(), 3)
        self.assertEqual(e(x), e1(x) * 21)
        self.assertNotEqual(e(x), 0)

        with np.errstate(all='ignore'):
            e = pints.SumOfErrors(
                [e4, e1, e1, e1, e1, e1], [10, 1, 1, 1, 1, 1])
            self.assertEqual(e.dimension(), 3)
            self.assertEqual(e(x), float('inf'))
            e = pints.SumOfErrors(
                [e4, e1, e1, e1, e1, e1], [0, 2, 0, 2, 0, 2])
            self.assertEqual(e.dimension(), 3)
            self.assertTrue(e(x), 6 * e1(x))
            e5 = pints.SumOfSquaresError(BadMiniProblem(float('-inf')))
            e = pints.SumOfErrors([e1, e5, e1], [2.1, 3.4, 6.5])
            self.assertTrue(np.isinf(e(x)))
            e = pints.SumOfErrors([e4, e5, e1], [2.1, 3.4, 6.5])
            self.assertTrue(np.isinf(e(x)))
            e5 = pints.SumOfSquaresError(BadMiniProblem(float('nan')))
            e = pints.SumOfErrors(
                [BadErrorMeasure(float('inf')), BadErrorMeasure(float('inf'))],
                [1, 1])
            self.assertEqual(e(x), float('inf'))
            e = pints.SumOfErrors(
                [BadErrorMeasure(float('inf')),
                 BadErrorMeasure(float('-inf'))],
                [1, 1])
            self.assertTrue(np.isnan(e(x)))
            e = pints.SumOfErrors(
                [BadErrorMeasure(5), BadErrorMeasure(float('nan'))], [1, 1])
            self.assertTrue(np.isnan(e(x)))
            e = pints.SumOfErrors([e1, e5, e1], [2.1, 3.4, 6.5])
            self.assertTrue(np.isnan(e(x)))
            e = pints.SumOfErrors([e4, e5, e1], [2.1, 3.4, 6.5])
            self.assertTrue(np.isnan(e(x)))

        # Wrong number of arguments
        self.assertRaises(ValueError, pints.SumOfErrors, [e1], [1])
        # Wrong argument types
        self.assertRaises(
            TypeError, pints.SumOfErrors, [e1, e1], [e1, 1])
        self.assertRaises(
            ValueError, pints.SumOfErrors, [e1, 3], [2, 1])
        # Mismatching sizes
        self.assertRaises(
            ValueError, pints.SumOfErrors, [e1, e1, e1], [1, 1])
        # Mismatching problem dimensions
        self.assertRaises(
            ValueError, pints.SumOfErrors, [e1, e1, e3], [1, 2, 3])


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
