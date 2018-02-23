#!/usr/bin/env python3
#
# Tests Pints io methods.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import print_function, unicode_literals
import os
import sys
import pints
import pints.io
import unittest


class TestIO(unittest.TestCase):
    """
    Tests Pints io methods.
    """

    def test_stream_capture(self):
        """
        Tests the StreamCapture class.
        """
        # Test stdout capture
        t1 = 'Hello everyone'
        t2 = 'How are you'
        tt = t1 + '\n' + t2
        with pints.io.StreamCapture() as c:
            print(t1)
            sys.stdout.write(t2)
        self.assertEqual(c.text(), tt)

        # Test stderr capture
        e1 = 'Oh no'
        e2 = 'What a terrible error'
        et = e1 + '\n' + e2
        with pints.io.StreamCapture(stdout=False, stderr=True) as c:
            print(e1, file=sys.stderr)
            sys.stderr.write(e2)
        self.assertEqual(c.text(), et)

        # Test double capture
        with pints.io.StreamCapture(stdout=True, stderr=True) as c:
            print(t1)
            print(e1, file=sys.stderr)
            sys.stderr.write(e2)
            sys.stdout.write(t2)
        self.assertEqual(c.text(), (tt, et))

    def test_temporary_directory(self):
        """
        Tests the temporary directory class.
        """
        with pints.io.TemporaryDirectory() as d:
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


if __name__ == '__main__':
    print('Add -v for more debug output')
    if '-v' in sys.argv:
        debug = True
    unittest.main()
