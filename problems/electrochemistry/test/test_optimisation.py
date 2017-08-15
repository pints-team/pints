#
# Tests the electrochemistry models (have to be compiled first!)
#
import unittest

DEFAULT = {
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
    'alpha': 0.53,
    }

class TestModel(unittest.TestCase):

    def test_simulation(self):
        """
        Runs a simple simulation
        """
        import electrochemistry
        # Create model
        model = electrochemistry.ECModel(DEFAULT)
        # Run simulation
        I,t = model.simulate()
        # Run simulation on specific time points
        I2, t2 = model.simulate(use_times=t)
        
        with open('./out.txt', 'w') as f:
            for time in t:
                f.write(str(time) + '\n')

    def fit(self, method):
        """
        Refits a model to its own data using the given method.
        """
        import pints
        import electrochemistry
        import numpy as np

        # Define wrapper around echem model
        class Model(pints.ForwardModel):
            def __init__(self):
                self._model = electrochemistry.ECModel(DEFAULT)
                self._parameters = ['E0', 'k0', 'Cdl']
            def dimension(self):
                return len(self._parameters)
            def simulate(self, parameters, times):
                self._model.set_params_from_vector(parameters,
                    self._parameters)
                current, t = self._model.simulate(use_times = times)
                return current

        # Create some toy data
        ecmodel = electrochemistry.ECModel(DEFAULT)
        values, times = ecmodel.simulate()

        # Test wrapper
        model = Model()
        # Get real parameter values
        # Note: Retrieving them from ECModel to get non-dimensionalised form!
        real = np.array([ecmodel.params[x] for x in model._parameters])
        # Test simulation via wrapper class
        values2 = model.simulate(real, times)
        self.assertEqual(len(values), len(values2))
        self.assertTrue(np.all(np.array(values) == np.array(values2)))

        # Set up problem
        problem = pints.SingleSeriesProblem(model, times, values)
        score = pints.SumOfSquaresError(problem)
        boundaries = pints.Boundaries(real*0.95, real*1.05)
        hint = real*0.99
        
        # Run optimisation
        x, f = method(score, boundaries, x0=hint)
        
        # Check result
        self.assertLess(f, 1e-9)

    def test_fit(self):
        """
        Runs an optimisation.
        """
        import pints
        #self.fit(pints.cmaes)
        #self.fit(pints.snes)
        self.fit(pints.xnes)
        #self.fit(pints.pso)
        

















