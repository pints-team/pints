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

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


debug = False


class TestEvaluators(unittest.TestCase):
    """
    Tests the evaluator classes and methods.
    """
    def __init__(self, name):
        super(TestEvaluators, self).__init__(name)

    def test_function(self):

        # Create test data
        xs = np.random.normal(0, 10, 10)
        ys = [f(x) for x in xs]

        # Test evaluate function
        self.assertTrue(np.all(ys == pints.evaluate(f, xs, parallel=True)))
        self.assertTrue(np.all(ys == pints.evaluate(f, xs, parallel=1)))
        self.assertTrue(np.all(ys == pints.evaluate(f, xs, parallel=False)))

    def test_sequential(self):

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
        e = pints.SequentialEvaluator(f_args, [10, 20])
        self.assertEqual(e.evaluate([1]), [31])

        # Args must be a sequence
        self.assertRaises(ValueError, pints.SequentialEvaluator, f_args, 1)

    def test_parallel(self):

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
        e = pints.SequentialEvaluator(f_args, [10, 20])
        self.assertEqual(e.evaluate([1]), [31])

        # Args must be a sequence
        self.assertRaises(ValueError, pints.ParallelEvaluator, f_args, args=1)

        # n-workers must be >0
        self.assertRaises(ValueError, pints.ParallelEvaluator, f, 0)

        # max tasks must be >0
        self.assertRaises(ValueError, pints.ParallelEvaluator, f, 1, 0)

        # Exceptions in called method should trigger halt, cause new exception
        e = pints.ParallelEvaluator(ioerror_on_five, n_workers=2)
        self.assertRaisesRegex(
            Exception, 'Exception in subprocess', e.evaluate, [1, 2, 5])
        e.evaluate([1, 2])

        # System exit
        e = pints.ParallelEvaluator(system_exit_on_four, n_workers=2)
        self.assertRaisesRegex(
            Exception, 'Exception in subprocess', e.evaluate, [1, 2, 4])
        e.evaluate([1, 2])

    def test_worker(self):
        """
        Manual test of worker, since cover doesn't pick up on its run method.
        """
        from pints._evaluation import _Worker as Worker

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

        w = Worker(
            interrupt_on_30, (), tasks, results, max_tasks, errors, error)
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

        w = Worker(
            interrupt_on_30, (), tasks, results, max_tasks, errors, error)
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

        w = Worker(
            interrupt_on_30, (), tasks, results, max_tasks, errors, error)
        w.run()

        self.assertEqual(results.get(timeout=0.01), (0, 2))
        self.assertTrue(results.empty())
        self.assertTrue(error.is_set())
        #self.assertFalse(errors.empty())   # Fails on travis!
        self.assertIsNotNone(errors.get(timeout=0.01))


def f(x):
    """
    Test function to parallelise. (Must be here so it can be pickled on
    windows).
    """
    return x ** 2


def f_args(x, y, z):
    return x + y + z


def ioerror_on_five(x):
    if x == 5:
        raise IOError
    return x


def interrupt_on_30(x):
    if x == 30:
        raise KeyboardInterrupt
    return 2 * x


def system_exit_on_four(x):
    if x == 4:
        raise SystemExit
    return x


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
