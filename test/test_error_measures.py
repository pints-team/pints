#!/usr/bin/env python3
#
# Tests the evaluator methods and classes.
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
        self.assertEqual(e([1, 2, 3]), -10)

    def test_root_mean_squared_error(self):
        p = MiniProblem()
        e = pints.RootMeanSquaredError(p)
        float(e([1, 2, 3]))
        self.assertEqual(e([-1, 2, 3]), 0)
        self.assertNotEqual(np.all(e([1, 2, 3])), 0)
        x = [0, 0, 0]
        y = np.sqrt((1 + 4 + 9) / 3)
        self.assertAlmostEqual(e(x), y)
        x = [1, 1, 1]
        y = np.sqrt((4 + 1 + 4) / 3)
        self.assertEqual(e(x), y)

    def test_sum_of_squares_error(self):
        p = MiniProblem()
        e = pints.SumOfSquaresError(p)
        float(e([1, 2, 3]))
        self.assertEqual(e([-1, 2, 3]), 0)
        self.assertNotEqual(np.all(e([1, 2, 3])), 0)
        x = [0, 0, 0]
        y = 1 + 4 + 9
        self.assertEqual(e(x), y)
        x = [1, 1, 1]
        y = 4 + 1 + 4
        self.assertEqual(e(x), y)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
