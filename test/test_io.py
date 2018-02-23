#!/usr/bin/env python3
#
# Tests the Pints io methods.
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
import numpy as np
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

    def test_load_save_samples(self):
        """
        Tests the load_samples and save_samples() methods.
        """

        m = 10  # 10 samples
        n = 5   # 5 parameters
        chain0 = np.random.uniform(size=(m, n))
        chain1 = np.random.uniform(size=(m, n))

        # Must support lists as well as arrays
        chain2 = []
        for row in np.random.uniform(size=(m, n)):
            chain2.append(list(row))

        # Check saving and loading
        with pints.io.TemporaryDirectory() as d:
            # Single chain
            filename = d.path('test.csv')
            pints.io.save_samples(filename, chain0)
            self.assertTrue(os.path.isfile(filename))
            test0 = pints.io.load_samples(filename)
            self.assertEqual(chain0.shape, test0.shape)
            self.assertTrue(np.all(chain0 == test0))
            self.assertFalse(chain0 is test0)

            # Multiple chains
            filename = d.path('multi.csv')
            pints.io.save_samples(filename, chain0, chain1, chain2)
            self.assertTrue(os.path.isfile(d.path('multi_0.csv')))
            self.assertTrue(os.path.isfile(d.path('multi_1.csv')))
            self.assertTrue(os.path.isfile(d.path('multi_2.csv')))
            test0, test1, test2 = pints.io.load_samples(filename, 3)
            self.assertEqual(chain0.shape, test0.shape)
            self.assertTrue(np.all(chain0 == test0))
            self.assertFalse(chain0 is test0)
            self.assertEqual(chain1.shape, test1.shape)
            self.assertTrue(np.all(chain1 == test1))
            self.assertFalse(chain1 is test1)
            self.assertEqual(np.asarray(chain2).shape, test2.shape)
            self.assertTrue(np.all(np.asarray(chain2) == test2))
            self.assertFalse(chain2 is test2)

            # Check invalid save_samples() calls
            self.assertRaises(ValueError, pints.io.save_samples, filename)
            chainX = np.random.uniform(size=(2,2,2))
            self.assertRaises(
                ValueError, pints.io.save_samples, filename, chainX)
            chainY = [[1, 2], [3, 4, 5]]
            self.assertRaises(
                ValueError, pints.io.save_samples, filename, chainY)

            # Test invalid load_samples calls
            filename = d.path('x.csv')
            try:
                self.assertRaises(
                    FileNotFoundError, pints.io.load_samples, filename)
                self.assertRaises(
                    FileNotFoundError, pints.io.load_samples, filename, 10)
            except NameError:
                self.assertRaises(
                    IOError, pints.io.load_samples, filename)
                self.assertRaises(
                    IOError, pints.io.load_samples, filename, 10)


if __name__ == '__main__':
    print('Add -v for more debug output')
    if '-v' in sys.argv:
        debug = True
    unittest.main()
