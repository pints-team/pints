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
        import time
        def ioerror_on_five(x):
            if x == 5:
                raise IOError
            return x

        e = pints.ParallelEvaluator(ioerror_on_five, n_workers=2)
        self.assertRaises(Exception, e.evaluate, range(10))
        try:
            e.evaluate([1, 2, 5])
        except Exception as ex:
            self.assertIn('Exception in subprocess', str(ex))
        e.evaluate([1, 2])

        # System exit
        def system_exit_on_four(x):
            if x == 4:
                raise SystemExit
            return x

        e = pints.ParallelEvaluator(system_exit_on_four, n_workers=2)
        self.assertRaises(Exception, e.evaluate, range(10))
        try:
            e.evaluate([1, 2, 4])
        except Exception as ex:
            self.assertIn('Exception in subprocess', str(ex))
        e.evaluate([1, 2])

        # Keyboard interrupt (Ctrl-C)
        def user_cancel_on_three(x):
            if x == 3:
                raise KeyboardInterrupt
            return x

        e = pints.ParallelEvaluator(user_cancel_on_three, n_workers=2)
        self.assertRaises(Exception, e.evaluate, range(10))
        try:
            e.evaluate([1, 2, 3])
        except Exception as ex:
            self.assertIn('Exception in subprocess', str(ex))
        e.evaluate([1, 2])

    def test_worker(self):
        """
        Manual test of worker, since cover doesn't pick up on its run method.
        """
        from pints._evaluation import _Worker as Worker

        # Define function
        def f(x):
            if x == 30:
                raise KeyboardInterrupt
            return 2 * x

        # Create queues for worker
        import multiprocessing
        tasks = multiprocessing.Queue()
        results = multiprocessing.Queue()
        errors = multiprocessing.Queue()
        error = multiprocessing.Event()
        tasks.put((0, 1))
        tasks.put((1, 2))
        tasks.put((2, 3))
        max_tasks = 3

        w = Worker(f, (), tasks, results, max_tasks, errors, error)
        w.run()

        self.assertEqual(results.get(timeout=0.01), (0, 2))
        self.assertEqual(results.get(timeout=0.01), (1, 4))
        self.assertEqual(results.get(timeout=0.01), (2, 6))
        self.assertTrue(results.empty())

        # Test worker stops if error flag is set
        tasks = multiprocessing.Queue()
        results = multiprocessing.Queue()
        errors = multiprocessing.Queue()
        error = multiprocessing.Event()
        tasks.put((0, 1))
        tasks.put((1, 2))
        tasks.put((2, 3))
        error.set()

        w = Worker(f, (), tasks, results, max_tasks, errors, error)
        w.run()

        self.assertEqual(results.get(timeout=0.01), (0, 2))
        self.assertTrue(results.empty())

        # Tests worker catches, stores and halts on exception
        tasks = multiprocessing.Queue()
        results = multiprocessing.Queue()
        errors = multiprocessing.Queue()
        error = multiprocessing.Event()
        tasks.put((0, 1))
        tasks.put((1, 30))
        tasks.put((2, 3))

        w = Worker(f, (), tasks, results, max_tasks, errors, error)
        w.run()

        self.assertEqual(results.get(timeout=0.01), (0, 2))
        self.assertTrue(results.empty())
        self.assertTrue(error.is_set())
        #self.assertFalse(errors.empty())   # Fails on travis!
        self.assertIsNotNone(errors.get(timeout=0.01))


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
