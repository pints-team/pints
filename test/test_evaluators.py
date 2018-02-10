#!/usr/bin/env python3
#
# Tests the evaluator methods and classes.
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import pints.toy as toy
import unittest
import numpy as np

debug = False


class TestEvaluators(unittest.TestCase):
    """
    Tests the evaluator classes and methods.
    """
    def __init__(self, name):
        super(TestEvaluators, self).__init__(name)

    def test_evaluators(self):

        # Create test function
        def f(x):
            return float(x)**2

        # Create test data
        xs = np.random.normal(0, 10, 100)
        ys = [f(x) for x in xs]

        # Test sequential evaluator
        e = pints.SequentialEvaluator(f)
        self.assertTrue(np.all(ys == e.evaluate(xs)))

        # Test sequential evaluator
        e = pints.ParallelEvaluator(f)
        self.assertTrue(np.all(ys == e.evaluate(xs)))

        # Test evaluate function
        self.assertTrue(np.all(ys == pints.evaluate(f, xs, parallel=True)))
        self.assertTrue(np.all(ys == pints.evaluate(f, xs, parallel=False)))



if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
