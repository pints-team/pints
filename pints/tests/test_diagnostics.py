#!/usr/bin/env python3
#
# Tests the basic methods of diagnostics.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import unittest
import pints
import numpy as np
import pints._diagnostics

class TestDiagnostics(unittest.TestCase):
    """
    Tests various diagnostic measures available in Pints
    """
    def test_autocorrelation(self):
        # Tests that autocorrelation measure is correct
        x = np.array([1, 2, 3, 4, -1, -1])
        y = pints._diagnostics.autocorrelation(x)
        y_true = np.array([1., 0.21354167, -0.41666667, -0.296875, -0.03645833,
                        0.03645833])
        
        for i in range(0, len(x)):
          self.assertTrue(np.abs(y[i] - y_true[i]) < 0.01)

if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
