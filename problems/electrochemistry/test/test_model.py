#!/usr/bin/env python
#
# Tests the electrochemistry models (have to be compiled first!)
#
import unittest
class TestModel(unittest.TestCase):
    def test_model(self):
        import pints
        import electrochemistry
        # Create model
        dim_params = {
            'reversed': True,
            'Estart': 0.5,
            'Ereverse': -0.1,
            'omega': 9.0152,
            'phase': 0,
            'dE': 0.08,
            'v': -0.08941,
            't_0': 0.001,
            'T': 297.0,
            'a': 0.07,
            'c_inf': 1*1e-3*1e-3,
            'D': 7.2e-6,
            'Ru': 8.0,
            'Cdl': 20.0*1e-6,
            'E0': 0.214,
            'k0': 0.0101,
            'alpha': 0.53
            }
        model = electrochemistry.ECModel(dim_params)
        # Run simulation
        I,t = model.simulate()

