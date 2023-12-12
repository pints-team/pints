#!/usr/bin/env python3
#
# Tests the evaluator methods and classes.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import multiprocessing
import numpy as np
import pints
import unittest


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

    def test_multi_sequential(self):

        # Create test data
        xs = np.random.normal(0, 10, 100)
        ys = [f(x) for x in xs]

        # Test sequential evaluator with multiple functions
        e = pints.MultiSequentialEvaluator([f for _ in range(100)])
        self.assertTrue(np.all(ys == e.evaluate(xs)))

        # check errors

        # not iterable
        with self.assertRaises(TypeError):
            e = pints.MultiSequentialEvaluator(3)

        # not callable
        with self.assertRaises(ValueError):
            e = pints.MultiSequentialEvaluator([f, 4])

        e = pints.MultiSequentialEvaluator([f for _ in range(100)])
        # Argument must be sequence
        with self.assertRaises(ValueError):
            e.evaluate(1)

        # wrong number of arguments
        with self.assertRaises(ValueError):
            e.evaluate([1 for _ in range(99)])

        # Test args
        e = pints.MultiSequentialEvaluator([f_args, f_args_plus1], [10, 20])
        self.assertEqual(e.evaluate([1, 1]), [31, 32])

        # Args must be a sequence
        self.assertRaises(
            ValueError, pints.MultiSequentialEvaluator, [f_args], 1)

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

    def test_parallel_random(self):
        # Test parallel processes get different random seed, but are
        # reproducible.

        # Ensure that worker processes don't all use the same random sequence
        # To test this, generate a random number in each task, and check that
        # the numbers don't match. With max-tasks-per-worker at 1, they should
        # all be the same without seeding.
        n = 20
        e = pints.ParallelEvaluator(
            random_int, n_workers=n, max_tasks_per_worker=1)
        x = np.array(e.evaluate([0] * n))
        self.assertFalse(np.all(x == x[0]))

        # Without max-tasks-per-worker, we still expect most workers to do 1
        # task, and some to do 2, maybe even three.
        e = pints.ParallelEvaluator(random_int, n_workers=n)
        x = set(e.evaluate([0] * n))
        self.assertTrue(len(x) > n // 2)

        # Getting the same numbers twice should be very unlikely
        x = np.array(e.evaluate([0] * n))
        y = np.array(e.evaluate([0] * n))
        #self.assertFalse(np.all(x) == np.all(y))
        self.assertTrue(len(set(x) | set(y)) > n // 2)

        # But with seeding we expect the same result twice
        np.random.seed(123)
        x = np.array(e.evaluate([0] * n))
        np.random.seed(123)
        y = np.array(e.evaluate([0] * n))
        self.assertTrue(np.all(x) == np.all(y))

        # Even with many more tasks than workers
        e = pints.ParallelEvaluator(random_int, n_workers=3)
        np.random.seed(123)
        x = np.array(e.evaluate([0] * 100))
        np.random.seed(123)
        y = np.array(e.evaluate([0] * 100))

    def test_worker(self):
        # Manual test of worker, since cover doesn't pick up on its run method.

        from pints._evaluation import _Worker as Worker

        # Create queues for worker
        tasks = multiprocessing.Queue()
        results = multiprocessing.Queue()
        errors = multiprocessing.Queue()
        error = multiprocessing.Event()
        tasks.put((0, 0, 1))
        tasks.put((1, 1, 2))
        tasks.put((2, 2, 3))
        max_tasks = 3
        max_threads = 1

        w = Worker(
            interrupt_on_30, (), tasks, results, max_tasks, max_threads,
            errors, error)
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
        tasks.put((0, 200, 1))
        tasks.put((1, 201, 2))
        tasks.put((2, 202, 3))
        error.set()

        w = Worker(
            interrupt_on_30, (), tasks, results, max_tasks, max_threads,
            errors, error)
        w.run()

        self.assertEqual(results.get(timeout=0.01), (0, 2))
        self.assertTrue(results.empty())

        # Tests worker catches, stores and halts on exception
        tasks = multiprocessing.Queue()
        results = multiprocessing.Queue()
        errors = multiprocessing.Queue()
        error = multiprocessing.Event()
        tasks.put((0, 100, 1))
        tasks.put((1, 400, 30))
        tasks.put((2, 200, 3))

        w = Worker(
            interrupt_on_30, (), tasks, results, max_tasks, max_threads,
            errors, error)
        w.run()

        self.assertEqual(results.get(timeout=0.01), (0, 2))
        self.assertTrue(results.empty())
        self.assertTrue(error.is_set())
        # \todo still relevant on GitHub actions?
        # self.assertFalse(errors.empty())   # Fails on travis!
        self.assertIsNotNone(errors.get(timeout=0.01))


def f(x):
    """
    Test function to parallelise. (Must be here so it can be pickled on
    windows).
    """
    return x ** 2


def f_args(x, y, z):
    return x + y + z


def f_args_plus1(x, y, z):
    return x + y + z + 1


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


def random_int(x):
    return np.random.randint(2**16)


if __name__ == '__main__':
    # Use 'spawn' method of process starting to prevent CI from hanging
    multiprocessing.set_start_method('spawn')
    unittest.main()
