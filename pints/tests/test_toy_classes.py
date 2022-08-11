#!/usr/bin/env python3
#
# Tests if the Toy ODE Model setters / getters work
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np
import pints
import pints.toy


class TestToyODEModel(unittest.TestCase):
    """
    Tests if the Toy ODE Model's non-abstract methods work.
    """

    def test_run(self):
        model = pints.toy.Hes1Model()
        vals = [1, 2, 3]
        model.set_initial_conditions(vals)
        self.assertTrue(np.array_equal(model.initial_conditions(),
                                       vals))


if __name__ == '__main__':
    unittest.main()
