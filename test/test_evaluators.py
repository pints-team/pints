#!/usr/bin/env python3
#
# Tests the evaluator methods and classes.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import unittest
import numpy as np

debug = False


class TestEvaluators(unittest.TestCase):
    """
    Tests the evaluator classes and methods.
    """
    def __init__(self, name):
        super(TestEvaluators, self).__init__(name)

    def test_function(self):

        # Create test function
        def f(x):
            return float(x)**2

        # Create test data
        xs = np.random.normal(0, 10, 100)
        ys = [f(x) for x in xs]

        # Test evaluate function
        self.assertTrue(np.all(ys == pints.evaluate(f, xs, parallel=True)))
        self.assertTrue(np.all(ys == pints.evaluate(f, xs, parallel=False)))

    def test_sequential(self):

        # Create test function
        def f(x):
            return float(x)**2

        # Create test data
        xs = np.random.normal(0, 10, 100)
        ys = [f(x) for x in xs]

        # Test sequential evaluator
        e = pints.SequentialEvaluator(f)
        self.assertTrue(np.all(ys == e.evaluate(xs)))
        self.assertRaises(ValueError, pints.SequentialEvaluator, 3)

        # Test args
        def f(x, y, z):
            self.assertEqual(y, 10)
            self.assertEqual(z, 20)

        e = pints.SequentialEvaluator(f, [10, 20])
        e.evaluate([1])

    def test_parallel(self):

        # Create test function
        def f(x):
            return float(x)**2

        # Create test data
        xs = np.random.normal(0, 10, 100)
        ys = [f(x) for x in xs]

        # Test parallel evaluator
        e = pints.ParallelEvaluator(f)
        self.assertTrue(np.all(ys == e.evaluate(xs)))

        # Function must be callable
        self.assertRaises(ValueError, pints.ParallelEvaluator, 3)

        # Test args
        def g(x, y, z):
            self.assertEqual(y, 10)
            self.assertEqual(z, 20)

        e = pints.ParallelEvaluator(g, args=[10, 20])
        e.evaluate([1])

        # Args must be a sequence
        self.assertRaises(ValueError, pints.ParallelEvaluator, g, args=1)

        # n-workers must be >0
        self.assertRaises(ValueError, pints.ParallelEvaluator, f, 0)

        # max tasks must be >0
        self.assertRaises(ValueError, pints.ParallelEvaluator, f, 1, 0)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
