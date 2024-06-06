#!/usr/bin/env python3
#
# Tests the test.shared methods.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import os
import sys
import unittest

import numpy as np

from shared import StreamCapture, TemporaryDirectory, UnitCircleBoundaries2D


class TestSharedTestModule(unittest.TestCase):
    """
    Tests the test.shared methods.
    """

    def test_stream_capture(self):
        # Tests the StreamCapture class.

        # Test stdout capture
        t1 = 'Hello everyone'
        t2 = 'How are you'
        tt = t1 + '\n' + t2
        with StreamCapture() as c:
            print(t1)
            sys.stdout.write(t2)
        self.assertEqual(c.text(), tt)

        # Test stderr capture
        e1 = 'Oh no'
        e2 = 'What a terrible error'
        et = e1 + '\n' + e2
        with StreamCapture(stdout=False, stderr=True) as c:
            print(e1, file=sys.stderr)
            sys.stderr.write(e2)
        self.assertEqual(c.text(), et)

        # Test double capture
        with StreamCapture(stdout=True, stderr=True) as c:
            print(t1)
            print(e1, file=sys.stderr)
            sys.stderr.write(e2)
            sys.stdout.write(t2)
        self.assertEqual(c.text(), (tt, et))

    def test_temporary_directory(self):
        # Tests the temporary directory class.

        with TemporaryDirectory() as d:
            # Test dir creation
            tempdir = d.path('')
            self.assertTrue(os.path.isdir(tempdir))

            # Test file creation
            text = 'Hello\nWorld'
            filename = d.path('test.txt')
            with open(filename, 'w') as f:
                f.write(text)
            with open(filename, 'r') as f:
                self.assertTrue(f.read() == text)
            self.assertTrue(os.path.isfile(filename))

            # Test invalid file creation
            self.assertRaises(ValueError, d.path, '../illegal.txt')

        # Test file and dir removal
        self.assertFalse(os.path.isfile(filename))
        self.assertFalse(os.path.isdir(tempdir))

        # Test runtime error when used outside of context
        self.assertRaises(RuntimeError, d.path, 'hello.txt')

    def test_unit_circle_boundaries_2d(self):
        # Tests the 2d unit circle boundaries used in composed boundaries
        # testing.
        c = UnitCircleBoundaries2D()
        self.assertEqual(c.n_parameters(), 2)
        self.assertTrue(c.check([0, 0]))
        self.assertTrue(c.check([0.5, 0]))
        self.assertTrue(c.check([-0.5, 0]))
        self.assertTrue(c.check([0, 0.5]))
        self.assertTrue(c.check([0, -0.5]))
        self.assertFalse(c.check([1, 0]))
        self.assertFalse(c.check([-1, 0]))
        self.assertFalse(c.check([0, 1]))
        self.assertFalse(c.check([0, -1]))
        self.assertTrue(c.check([1 - 1e-12, 0]))
        self.assertTrue(c.check([-1 + 1e-12, 0]))
        self.assertTrue(c.check([0, 1 - 1e-12]))
        self.assertTrue(c.check([0, -1 + 1e-12]))
        x, y = np.cos(0.123), np.sin(0.123)
        self.assertFalse(c.check([x, y]))
        xs = c.sample(100)
        self.assertEqual(xs.shape, (100, 2))
        for i, x in enumerate(xs):
            self.assertTrue(c.check(x))

        c = UnitCircleBoundaries2D(-5, 2)
        self.assertEqual(c.n_parameters(), 2)
        self.assertFalse(c.check([0, 0]))
        self.assertTrue(c.check([-5, 2]))
        self.assertTrue(c.check([-5, 3 - 1e-12]))
        for i, x in enumerate(c.sample(10)):
            self.assertTrue(c.check(x))

        self.assertRaises(Exception, c.check, [0, 0, 0])
        self.assertRaises(Exception, c.check, [0])


if __name__ == '__main__':
    unittest.main()
