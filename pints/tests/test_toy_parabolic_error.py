#!/usr/bin/env python
#
# Tests the parabolic error toy error measure.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
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
