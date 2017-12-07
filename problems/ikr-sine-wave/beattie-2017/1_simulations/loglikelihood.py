#!/usr/bin/env python2
#
# Show Kylie's model, for cell 5
#
# Compare with her simulation and score
# 
from __future__ import division
from __future__ import print_function
import os
import sys
import pints
import numpy as np
import myokit
import myokit.pacing as pacing

root = os.path.realpath(os.path.join('..', '..'))

cell = 5
model_file = os.path.join(root, 'models', 'beattie-2017-ikr-markov.mmt')
data_file = os.path.join(root, 'sine-wave-data',  
    'cell-' + str(cell) + '.csv')

# Load Kylie's data
kylie_sim = np.loadtxt('figure_4_model_fit_new_model_fit_tol10.txt')
kylie_sim = kylie_sim[:,1]
kylie_pro = np.loadtxt('figure_4_model_fit_protocol.txt')
kylie_tim = kylie_pro[:,0] * 1000
kylie_pro = kylie_pro[:,1]

#
# Cell temperature
#
temperatures = {
    5 : 21.4,   # 16713110
    }
#if strcmp(exp_ref,'16708016')==1    temperature = 21.8;
#if strcmp(exp_ref,'16708060')==1    temperature = 21.7;
#if strcmp(exp_ref,'16704047')==1    temperature = 21.6;
#if strcmp(exp_ref,'16704007')==1    temperature = 21.2;
#if strcmp(exp_ref,'16713003')==1    temperature = 21.3;
#if strcmp(exp_ref,'16715049')==1    temperature = 21.4;
#if strcmp(exp_ref,'16707014')==1    temperature = 21.4;
#if strcmp(exp_ref,'16708118')==1    temperature = 21.7;
temperature = temperatures[cell]

#
# Guesses for lower conductance
#
lower_conductances = {
    5: 0.0612,  # 16713110
    }
#if strcmp(exp_ref,'16708118')==1    lower_conductance = 0.0170;
#if strcmp(exp_ref,'16704047')==1    lower_conductance = 0.0434;
#if strcmp(exp_ref,'16704007')==1    lower_conductance = 0.0886;
#if strcmp(exp_ref,'16707014')==1    lower_conductance = 0.0203;
#if strcmp(exp_ref,'16708060')==1    lower_conductance = 0.0305;
#if strcmp(exp_ref,'16708016')==1    lower_conductance = 0.0417;
#if strcmp(exp_ref,'16713003')==1    lower_conductance = 0.0478;
#if strcmp(exp_ref,'16715049')==1    lower_conductance = 0.0255;
#if strcmp(exp_ref,'average')==1     lower_conductance = 0.0410;
lower_conductance = lower_conductances[cell]
upper_conductance = 10 * lower_conductance

#
# Guesses for alpha/beta parameter bounds
#
lower_alpha = 1e-7
upper_alpha = 1e0      #1e3
lower_beta  = 1e-7
upper_beta  = 0.4

#
# Load data
#
log = myokit.DataLog.load_csv(data_file).npview()
times = log.time()
current = log['current']
voltage = log['voltage']
del(log)

#
# Estimate noise from start of data
# Kylie uses the first 200ms, where I = 0 + noise
#
sigma_noise = np.std(current[:2000])

#
# Protocol info
#
dt = 0.1
steps = [
    [-80, 250.1],
    [-120, 50],
    [-80, 200],
    [40, 1000],
    [-120, 500],
    [-80, 1000],
    [-30, 3500],
    [-120, 500],
    [-80, 1000],
    ]

#
# Create capacitance filter based on protocol
#
cap_duration = 5 # Same as Kylie (also, see for example step at 500ms)
fcap = np.ones(len(current), dtype=int)
if True:
    # Skip this bit to show/use all data
    offset = 0
    for f, t in steps[:-1]:
        offset += t
        i1 = int(offset / dt)
        i2 = i1 + int(cap_duration / dt)
        fcap[i1:i2] = 0
fcap = fcap > 0

#
# Apply capacitance filter to data
#
times = times[fcap]
voltage = voltage[fcap]
current = current[fcap]
kylie_tim = kylie_tim[fcap]
kylie_sim = kylie_sim[fcap]
kylie_pro = kylie_pro[fcap]

#
# Calculate reversal potential like Kylie does
#
def erev(temperature):
    T = 273.15 + temperature
    F = 96485.0
    R = 8314.0
    K_i = 130.0
    k_o = 4.0
    return ((R*T)/F) * np.log(k_o/K_i)
E = erev(temperature)

#
# Create ForwardModel
#
class ModelWithProtocol(pints.ForwardModel):
    parameters = [
        'ikr.p1',
        'ikr.p2',
        'ikr.p3',
        'ikr.p4',
        'ikr.p5',
        'ikr.p6',
        'ikr.p7',
        'ikr.p8',
        'ikr.p9',
        ]
    def __init__(self):
        # Load model
        model = myokit.load_model(model_file)
        # Set reversal potential
        model.get('nernst.EK').set_rhs(E)
        # Add sine-wave equation to model
        model.get('membrane.V').set_rhs(
            'if(engine.time >= 3000.1 and engine.time < 6500.1,'
            + ' - 30'
            + ' + 54 * sin(0.007 * (engine.time - 2500.1))'
            + ' + 26 * sin(0.037 * (engine.time - 2500.1))'
            + ' + 10 * sin(0.190 * (engine.time - 2500.1))'
            + ', engine.pace)')
        # Create step protocol
        protocol = myokit.Protocol()
        for f, t in steps:
            protocol.add_step(f, t)
        # Create simulation
        self.simulation = myokit.Simulation(model, protocol)
        # Set Kylie tolerances
        #self.simulation.set_tolerance(1e-8, 1e-8)
        self.simulation.set_tolerance(1e-10, 1e-10)
    def dimension(self):
        return len(self.parameters)
    def simulate(self, parameters, times):
        # Note: Kylie doesn't do pre-pacing!
        # Update model parameters
        for i, name in enumerate(self.parameters):
            self.simulation.set_constant(name, parameters[i])
        # Run
        self.simulation.reset()
        try:
            d = self.simulation.run(
                np.max(times+0.5*dt),
                log_times = times,
                log = ['engine.time', 'ikr.IKr', 'membrane.V'],
                ).npview()
        except myokit.SimulationError:
            return times * float('inf')
        # Store membrane potential for debugging
        self.simulated_v = d['membrane.V']
        
        # Apply capacitance filter and return
        return d['ikr.IKr']

class ModelDataClamp(pints.ForwardModel):
    parameters = [
        'ikr.p1',
        'ikr.p2',
        'ikr.p3',
        'ikr.p4',
        'ikr.p5',
        'ikr.p6',
        'ikr.p7',
        'ikr.p8',
        'ikr.p9',
        ]
    def __init__(self):
        # Load model
        model = myokit.load_model(model_file)
        # Set reversal potential to Kylie value
        model.get('nernst.EK').set_rhs(E)
        # Create simulation
        self.simulation = myokit.Simulation(model)
        # Apply data-clamp
        self.simulation.set_fixed_form_protocol(times, voltage)
        # Set Kylie tolerances
        self.simulation.set_tolerance(1e-8, 1e-8)
        # Set Kylie max step size (also needed for data clamp)
        self.simulation.set_max_step_size(0.1)
    def dimension(self):
        return len(self.parameters)
    def simulate(self, parameters, times):
        # Update model parameters
        for i, name in enumerate(self.parameters):
            self.simulation.set_constant(name, parameters[i])
        # Run
        self.simulation.reset()
        try:
            d = self.simulation.run(
                np.max(times),
                log_times = times,
                log = ['ikr.IKr', 'membrane.V'],
                ).npview()
        except myokit.SimulationError:
            return times * float('inf')
        # Store membrane potential for debugging
        self.simulated_v = d['membrane.V']
        # Apply capacitance filter and return
        return d['ikr.IKr']

#
# Choose simulation implementation
#
model = ModelWithProtocol()
#model = ModelDataClamp()

#
# Define problem
#
problem = pints.SingleSeriesProblem(model, times, current)

#
# Select a score function
#
score = pints.SumOfSquaresError(problem)

#
# Load kylie's result
#
filename = 'kylies-solution.txt'
with open(filename, 'r') as f:
    obtained_parameters = [float(x) for x in f.readlines()]
obtained_score = score(obtained_parameters)

#
# Show equivalent log-likelihood with estimated std of noise
#
log_likelihood = pints.KnownNoiseLogLikelihood(problem, sigma_noise)

print('Myokit/Pints version:')
print('  Log-likelihood: '
    + pints.strfloat(log_likelihood(obtained_parameters)))

#
# Look at log-likelihood for my signal, compare with that for kylie's signal
# 

def loglikelihood(x, y, sigma):
    """ Calculate the log-likelihood of two signals matching """
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    return (
        -0.5 * n * np.log(2 * np.pi)
        -n * np.log(sigma)
        -np.sum((x - y)**2) / (2 * sigma**2)
        )

simulated = model.simulate(obtained_parameters, times)
print('Myokit signal comparison:')
print('  Log-likelihood: '
    + pints.strfloat(loglikelihood(simulated, current, sigma_noise)))
    
print('Matlab signal comparison:')
print('  Log-likelihood: '
    + pints.strfloat(loglikelihood(kylie_sim, current, sigma_noise)))


