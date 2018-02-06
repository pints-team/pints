#!/usr/bin/env python3
#
# Tests if the Hodgkin-Huxley IK (toy) problem works.
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import unittest
import numpy as np
import pints
import pints.toy


class TestHodgkinHuxleyIKProblem(unittest.TestCase):
    """
    Tests if the Hodgkin-Huxley IK (toy) problem works.
    """

    def test_start_with_zero(self):
        problem = pints.toy.HodgkinHuxleyIKProblem()
        p0 = problem.suggested_parameters()
        self.assertEqual(len(p0), problem.dimension())
        times = problem.times()
        self.assertTrue(np.all(times[1:] > times[:-1]))
        values = problem.evaluate(p0)
        self.assertEqual(len(times), len(values))
        folded = problem.fold(times, values)
        self.assertEqual(len(folded), 12)
        # Test protocol out of bounds
        self.assertEqual(problem._protocol(-1), -75)
        self.assertEqual(problem._protocol(9999), -75)


if __name__ == '__main__':
    unittest.main()
