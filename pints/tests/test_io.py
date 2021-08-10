#!/usr/bin/env python3
#
# Tests the Pints io methods.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import os
import pints
import pints.io
import numpy as np
import unittest

from shared import TemporaryDirectory


class TestIO(unittest.TestCase):
    """
    Tests Pints io methods.
    """

    def test_load_save_samples(self):
        # Tests the load_samples and save_samples() methods.

        m = 10  # 10 samples
        n = 5   # 5 parameters
        chain0 = np.random.uniform(size=(m, n))
        chain1 = np.random.uniform(size=(m, n))

        # Must support lists as well as arrays
        chain2 = []
        for row in np.random.uniform(size=(m, n)):
            chain2.append(list(row))

        # Check saving and loading
        with TemporaryDirectory() as d:
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
            self.assertRaisesRegex(
                ValueError, 'At least one set of samples',
                pints.io.save_samples, filename)
            chainX = np.random.uniform(size=(2, 2, 2))
            self.assertRaisesRegex(
                ValueError, 'must be given as 2d arrays',
                pints.io.save_samples, filename, chainX)
            chainY = [[1, 2], [3, 4, 5]]
            self.assertRaisesRegex(
                ValueError, 'same length',
                pints.io.save_samples, filename, chainY)

            # Test invalid load_samples calls
            self.assertRaisesRegex(
                ValueError, 'integer greater than zero',
                pints.io.load_samples, filename, 0)
            filename = d.path('x.csv')
            self.assertRaises(
                FileNotFoundError, pints.io.load_samples, filename)
            self.assertRaises(
                FileNotFoundError, pints.io.load_samples, filename, 10)


if __name__ == '__main__':
    unittest.main()
