#
# Tests the electrochemistry models (have to be compiled first!)
#
import unittest
import math

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

DEFAULT_POMS = {
    'reversed': False,
    'Estart': 0.6,
    'Ereverse': -0.1,
    #'omega': 60.05168,
    'omega': 6.05168,
    'phase': 0,
    'dE': 20e-3,
    'v': -0.1043081,
    't_0': 0.00,
    'T': 298.2,
    'a': 0.0707,
    'c_inf': 0.1*1e-3*1e-3,
    'Ru': 50.0,
    'Cdl': 0.000008,
    'Gamma' : 0.7*53.0e-12,
    'alpha1': 0.5,
    'alpha2': 0.5,
    'E01': 0.368,
    'E02': 0.338,
    'E11': 0.227,
    'E12': 0.227,
    'E21': 0.011,
    'E22': -0.016,
    'k01': 7300,
    'k02': 7300,
    'k11': 1e4,
    'k12': 1e4,
    'k21': 2500,
    'k22': 2500
    }

class TestModel(unittest.TestCase):
    def test_unwrapped(self):
        """
        Runs a simple simulation
        """
        import electrochemistry
        import numpy as np
        # Create model
        model = electrochemistry.ECModel(DEFAULT)
        # Run simulation
        values, times = model.simulate()
        # Run simulation on specific time points
        values2, times2 = model.simulate(use_times=times)

        self.assertEqual(len(values), len(values2))
        self.assertTrue(np.all(np.array(values) == np.array(values2)))

    def test_wrapper(self):
        """
        Wraps a `pints.ForwardModel` around a model.
        """
        import pints
        import electrochemistry
        import numpy as np

        # Create some toy data
        ecmodel = electrochemistry.ECModel(DEFAULT)
        values, times = ecmodel.simulate()

        # Test wrapper
        parameters = ['E0', 'k0', 'Cdl']
        pints_model = electrochemistry.PintsModelAdaptor(ecmodel,parameters)

        # Get real parameter values
        # Note: Retrieving them from ECModel to get non-dimensionalised form!
        real = np.array([ecmodel.params[x] for x in parameters])
        # Test simulation via wrapper class
        values2 = pints_model.simulate(real, times)
        self.assertEqual(len(values), len(values2))
        self.assertTrue(np.all(np.array(values) == np.array(values2)))

    def test_poms_with_bayesian_loglike(self):
        import pints
        import electrochemistry
        import numpy as np

        poms_model = electrochemistry.POMModel(DEFAULT_POMS)
        values, times = poms_model.simulate()

        names = ['E01','E02','E11','E12','E21','E22',
                 'k01','k02','k11','k12','k21','k22','gamma']

        e0_buffer = 0.1*(poms_model.params['Estart'] - poms_model.params['Ereverse'])
        max_current = np.max(values)
        max_k0 = poms_model.non_dimensionalise(10000,'k01')
        lower_bounds = [poms_model.params['Ereverse']+e0_buffer,
                poms_model.params['Ereverse']+e0_buffer,
                poms_model.params['Ereverse']+e0_buffer,
                poms_model.params['Ereverse']+e0_buffer,
                poms_model.params['Ereverse']+e0_buffer,
                poms_model.params['Ereverse']+e0_buffer,
                0,
                0,
                0,
                0,
                0,
                0,
                0.1,
                0.005*max_current]

        upper_bounds = [poms_model.params['Estart']-e0_buffer,
                poms_model.params['Estart']-e0_buffer,
                poms_model.params['Estart']-e0_buffer,
                poms_model.params['Estart']-e0_buffer,
                poms_model.params['Estart']-e0_buffer,
                poms_model.params['Estart']-e0_buffer,
                max_k0,
                max_k0,
                max_k0,
                max_k0,
                max_k0,
                max_k0,
                5,
                0.03*max_current]

        priors = []
        E0 = 0.5*(poms_model.params['E01'] + poms_model.params['E02'])
        E0_diff = 1
        priors.append(pints.NormalPrior(E0,(2*E0_diff)**2))
        priors.append(pints.NormalPrior(E0,(2*E0_diff)**2))
        E1 = 0.5*(poms_model.params['E11'] + poms_model.params['E12'])
        E1_diff = 1
        priors.append(pints.NormalPrior(E1,(2*E1_diff)**2))
        priors.append(pints.NormalPrior(E1,(2*E1_diff)**2))
        E2 = 0.5*(poms_model.params['E21'] + poms_model.params['E22'])
        E2_diff = 1
        priors.append(pints.NormalPrior(E2,(2*E2_diff)**2))
        priors.append(pints.NormalPrior(E2,(2*E2_diff)**2))
        priors.append(pints.UniformPrior(lower_bounds[6:14],upper_bounds[6:14]))

        # Load a forward model
        pints_model = electrochemistry.PintsModelAdaptor(poms_model,names)

        # Create an object with links to the model and time series
        problem = pints.SingleSeriesProblem(pints_model, times, values)

        # Create a log-likelihood function scaled by n
        log_likelihood = pints.ScaledLogLikelihood(pints.UnknownNoiseLogLikelihood(problem))

        # Create a uniform prior over both the parameters and the new noise variable
        prior = pints.ComposedPrior(*priors)

        # Create an unnormalised (prior * likelihood)
        score = pints.LogPosterior(prior, log_likelihood)

        # Select some boundaries
        boundaries = pints.Boundaries(lower_bounds,upper_bounds)

        # pick a reasonable estimate
        x0 = [E0,E0,E1,E1,E2,E2] \
            + [0.5*(u-l) for l,u in zip(lower_bounds[6:14],upper_bounds[6:14])]

        self.assertAlmostEqual(log_likelihood(x0)+math.log(prior(x0)), score(x0))



if __name__ == '__main__':
    unittest.main()
