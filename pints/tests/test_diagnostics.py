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
          
    def test_autocorrelation_negative(self):
        # Tests autocorrelation_negative yields the correct result
        # under both possibilities
        
        # Test for case where there is a negative element
        x = np.array([1, 2, 3, 4, -1, -1])
        self.assertTrue(pints._diagnostics.autocorrelate_negative(x) == 4)
        
        # Test for case with no negative elements
        x = np.array([1, 2, 3, 4, 1, 1])
        self.assertTrue(pints._diagnostics.autocorrelate_negative(x) == 7)
    
    def test_ess_single_param(self):
        # Tests that ESS for a single parameter is correct
        
        # For case with negative elements in x
        x = np.array([1, 2, 3, 4, -1, -1])
        self.assertTrue(np.abs(pints._diagnostics.autocorrelate_negative(x) - 1.75076) < 0.01)
        
        # Case with positive elements only in x
        x = np.array([1, 2, 3, 4, 1, 1])
        self.assertTrue(np.abs(pints._diagnostics.autocorrelate_negative(x) - 
                               1.846154) < 0.01)
        
    def test_effective_sample_size(self):
        # Tests ess for a matrix of parameters
        
        # matrix with two columns of samples
        x = np.transpose(np.array([[1.0, 1.1, 1.4, 1.3, 1.3],
                                   [1.0, 2.0, 3.0, 4.0, 5.0]]))
        y = pints._diagnostics.effective_sample_size(x)
        self.assertTrue(np.abs(y[0] - 1.439232) < 0.01)
        self.assertTrue(np.abs(y[1] - 1.315789) < 0.01)
        
    def test_within(self):
        # Tests within chain variance calculation
        
        # matrix with two columns of samples
        x = np.array([[1.0, 1.1, 1.4, 1.3, 1.3],
                                   [1.0, 2.0, 3.0, 4.0, 5.0]])
        self.assertTrue(np.abs(pints._diagnostics.within(x) - 1.2635) < 0.01)

if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
