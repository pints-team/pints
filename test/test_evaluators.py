#!/usr/bin/env python
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

        # Function must be callable
        self.assertRaises(ValueError, pints.SequentialEvaluator, 3)

        # Argument must be sequence
        self.assertRaises(ValueError, e.evaluate, 1)

        # Test args
        def g(x, y, z):
            self.assertEqual(y, 10)
            self.assertEqual(z, 20)

        e = pints.SequentialEvaluator(g, [10, 20])
        e.evaluate([1])

        # Args must be a sequence
        self.assertRaises(ValueError, pints.SequentialEvaluator, g, 1)

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

        # Argument must be sequence
        self.assertRaises(ValueError, e.evaluate, 1)

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

        # Exceptions in called method should trigger halt, cause new exception

        # Any old exception
        def ioerror_on_fifty(x):
            if x == 50:
                raise IOError
            return x

        e = pints.ParallelEvaluator(ioerror_on_fifty)
        self.assertRaises(Exception, e.evaluate, range(100))
        try:
            e.evaluate([1, 2, 50])
        except Exception as e:
            self.assertIn('Exception in subprocess', str(e))

        # System exit
        def system_exit_on_40(x):
            if x == 40:
                raise SystemExit
            return x

        e = pints.ParallelEvaluator(ioerror_on_fifty)
        self.assertRaises(Exception, e.evaluate, range(100))
        try:
            e.evaluate([1, 2, 40])
        except Exception as e:
            self.assertIn('Exception in subprocess', str(e))

        # Keyboard interrupt (Ctrl-C)
        def user_cancel_on_30(x):
            if x == 30:
                raise KeyboardInterrupt
            return x

        e = pints.ParallelEvaluator(ioerror_on_fifty)
        self.assertRaises(Exception, e.evaluate, range(100))
        try:
            e.evaluate([1, 2, 30])
        except Exception as e:
            self.assertIn('Exception in subprocess', str(e))


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
