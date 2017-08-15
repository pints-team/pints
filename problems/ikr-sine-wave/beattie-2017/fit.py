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

model_file = os.path.realpath(os.path.join('..', 'models', 
    'beattie-2017-ikr.mmt'))
data_file = os.path.realpath(os.path.join('..', 'sine-wave-data',
    'sine-wave.csv'))

#
# Load data
#
log = myokit.DataLog.load_csv(data_file).npview()
times = log.time()
values = log['current']

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
fcap = np.ones(len(values), dtype=int)
offset = 0
for f, t in steps[:-1]:
    offset += t
    i1 = int(offset / dt)
    i2 = i1 + int(cap_duration / dt)
    fcap[i1:i2] = 0

#
# Apply capacitance filter to data
#
values = values * fcap

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
            return float('nan')
        # Apply capacitance filter and return
        return d['ikr.IKr'] * fcap
model = Model()

#
# Define problem
#
problem = pints.SingleSeriesProblem(model, times, values)

#
# Select a score function
#
score = pints.SumOfSquaresError(problem)

#
# Set up boundaries
#
lower = [1e-7] * 8 + [1e-5]
upper = [1e-1] * 8 + [1e3]
boundaries = pints.Boundaries(lower, upper)

#
# Run an optimisation
#
with np.errstate(all='ignore'): # Tell numpy not to issue warnings
    obtained_parameters, obtained_score = pints.cmaes(
        score,
        boundaries,
        )

print('Obtained parameters:' )
for x in obtained_parameters:
    print(pints.strfloat(x))


