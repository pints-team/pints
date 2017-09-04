#!/usr/bin/env python2
#
# Fit Kylie's model to real data
# 
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join('..', '..', '..')))
import pints
sys.path.append(os.path.abspath(os.path.join('..', 'myokit')))
import myokit
import myokit.pacing as pacing

cell = 5
model_file = os.path.realpath(os.path.join('..', 'models', 
    'beattie-2017-ikr.mmt'))
data_file = os.path.realpath(os.path.join('..', 'sine-wave-data', 
    'cell-' + str(cell) + '.csv'))

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
#TODO: Use temperature?

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
#
sigma_noise = np.std(current[:2000])

#
# Protocol info
#
dt = 0.1
steps = [
    (-80, 250),
    (-120, 50),
    (-80, 200),
    (40, 1000),
    (-120, 500),
    (-80, 1000),
    (-30, 3500),
    (-120, 500),
    (-80, 1000),
    ]

#
# Create capacitance filter based on protocol
#
cap_duration = 1.5
fcap = np.ones(len(current), dtype=int)
offset = 0
for f, t in steps[:-1]:
    offset += t
    i1 = int(offset / dt)
    i2 = i1 + int(cap_duration / dt)
    fcap[i1:i2] = 0

#
# Apply capacitance filter to data
#
current = current * fcap

#
# Create ForwardModel
#
class Model(pints.ForwardModel):
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
        # Create pre-pacing protocol
        protocol = myokit.pacing.constant(-80)
        # Create pre-pacing simulation
        self.simulation1 = myokit.Simulation(model, protocol)
        # Add sine-wave equation to model
        model.get('membrane.V').set_rhs(
            'if(engine.time < 3000 or engine.time >= 6500,'
            + ' engine.pace, '
            + ' - 30'
            + ' + 54 * sin(0.007 * (engine.time - 2500))'
            + ' + 26 * sin(0.037 * (engine.time - 2500))'
            + ' + 10 * sin(0.190 * (engine.time - 2500))'
            + ')')
        # Create step protocol
        protocol = myokit.Protocol()
        for f, t in steps:
            protocol.add_step(f, t)
        # Create simulation for sine-wave protocol
        self.simulation2 = myokit.Simulation(model, protocol)
        #self.simulation2.set_tolerance(1e-8, 1e-8)
    def dimension(self):
        return len(self.parameters)
    def simulate(self, parameters, times):
        # Update model parameters
        for i, name in enumerate(self.parameters):
            self.simulation1.set_constant(name, parameters[i])
            self.simulation2.set_constant(name, parameters[i])
        # Run
        self.simulation1.reset()
        self.simulation2.reset()
        try:
            self.simulation1.pre(10000)
            self.simulation2.set_state(self.simulation1.state())
            d = self.simulation2.run(
                np.max(times),
                log_times = times,
                log = ['ikr.IKr'],
                ).npview()
        except myokit.SimulationError:
            return float('inf')
        # Apply capacitance filter and return
        return d['ikr.IKr'] * fcap
model = Model()

#
# Define problem
#
problem = pints.SingleSeriesProblem(model, times, current)

#
# Select a score function
#
score = pints.SumOfSquaresError(problem)

#
# Set up boundaries
#
lower = [
    lower_alpha,
    lower_beta,
    lower_alpha,
    lower_beta,
    lower_alpha,
    lower_beta,
    lower_alpha,
    lower_beta,
    lower_conductance,
    ]
upper = [
    upper_alpha,
    upper_beta,    
    upper_alpha,
    upper_beta,    
    upper_alpha,
    upper_beta,    
    upper_alpha,
    upper_beta,    
    upper_conductance,
    ]
boundaries = pints.Boundaries(lower, upper)

#
# Run, or load earlier result
#
if len(sys.argv) > 1 and sys.argv[1] == 'show':
    
    #
    # Load earlier result
    #
    filename = 'last-solution.txt'
    if len(sys.argv) > 2:
        filename = sys.argv[2]
    with open(filename, 'r') as f:
        obtained_parameters = [float(x) for x in f.readlines()]
    obtained_score = score(obtained_parameters)

else:
    #
    # Run an optimisation
    #
    with np.errstate(all='ignore'): # Tell numpy not to issue warnings
        obtained_parameters, obtained_score = pints.cmaes(
            score,
            boundaries,
            )

    #
    # Store result
    #
    with open('last-solution.txt', 'w') as f:
        for x in obtained_parameters:
            f.write(pints.strfloat(x) + '\n')

#
# Show obtained parameters and score
#
print('Obtained parameters:')
for x in obtained_parameters:
    print(pints.strfloat(x))
print('Final score:')
print(obtained_score)

#
# Show equivalent log-likelihood with estimated std of noise
#
log_likelihood = pints.KnownNoiseLogLikelihood(problem, sigma_noise)
print('Sigma noise: ' + str(sigma_noise))
print('Log-likelihood: ' + pints.strfloat(log_likelihood(obtained_parameters)))

#
# Show result
#
if len(sys.argv) > 1 and sys.argv[1] == 'show':    

    # Simulate
    simulated = model.simulate(obtained_parameters, times)

    # Plot
    import matplotlib.pyplot as pl
    pl.figure()
    pl.subplot(2,1,1)
    pl.plot(times, voltage)
    pl.subplot(2,1,2)
    pl.plot(times, current, label='real')
    pl.plot(times, simulated, label='fit')
    pl.legend(loc='lower right')
    pl.show()
    
    

