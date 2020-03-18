#!/usr/bin/env python
#
# Tests the parabolic error toy error measure.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.toy
import unittest


class TestParabolicError(unittest.TestCase):
    """
    Tests the parabolic error toy error measure.
    """
    def test_parabolic_error(self):

        # Test basics
        f = pints.toy.ParabolicError()
        self.assertEqual(f.n_parameters(), 2)
        self.assertEqual(list(f.optimum()), [0, 0])
        self.assertEqual(f([0, 0]), 0)
        self.assertTrue(f([0.1, 0.1]) > 0)

        f = pints.toy.ParabolicError([1, 1, 1])
        self.assertEqual(f.n_parameters(), 3)
        self.assertEqual(list(f.optimum()), [1, 1, 1])
        self.assertEqual(f([1, 1, 1]), 0)
        self.assertTrue(f([1.1, 1.1, 1.1]) > 0)


if __name__ == '__main__':
    unittest.main()
