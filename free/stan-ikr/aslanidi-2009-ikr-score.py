#!/usr/bin/env python
import myokit
import myokit.pacing as pacing
import numpy as np
import os

#
# Score function for the Aslanidi model, using the toy-data protocol
# 
#
#

def create_toy_data_protocol():
    """
    Returns the protocol used to generate the toy data (as a myokit.Protocol
    object).
    """
    tpre  = 2000        # Time before step to variable V
    tstep = 5000        # Time at variable V
    tpost = 3000        # Time after step to variable V
    ttotal = tpre + tstep + tpost
    vhold = -80
    vmin = -100
    vmax = 50
    vres = 10           # Difference in V between steps
    v = np.arange(vmin, vmax + vres, vres)
    p = pacing.steptrain(
            vsteps=v,
            vhold=vhold,
            tpre=tpre,
            tstep=tstep,
            tpost=tpost)
    return p

def pre_pace_model(model):
    """
    Pre-treats the model to get the same initial state as used in the toy data
    generation.
    """
    vhold = -80
    p = pacing.constant(vhold)
    s = myokit.Simulation(model, p)
    s.run(10000, log=myokit.LOG_NONE)
    model.set_state(s.state())

# Load model and pre-pace
model = myokit.load_model('aslanidi-2009-ikr.mmt')
pre_pace_model(model)

# Load protocol, get protocol duration
protocol = create_toy_data_protocol()
duration = protocol.characteristic_time()

# Load toy data, and wrap numpy arrays around each series
real = myokit.DataLog.load_csv('aslanidi-2009-ikr-with-noise.csv')
real = real.npview()

# Define parameters
parameters = [
    'ikr.p1',
    'ikr.p2',
    'ikr.p3',
    'ikr.p4',
    'ikr.p5',
    'ikr.p6',
    'ikr.p7',
    'ikr.p8',
    ]
    
# Get real parameter values from model
real_values = [model.get(name).eval() for name in parameters]
print('True solution:')
print(real_values)
    
# define score function (sum of squares)
simulation = myokit.Simulation(model, protocol)
def score(p):
    simulation.reset()
    for i, name in enumerate(parameters):
        simulation.set_constant(name, p[i])
    try:
        data = simulation.run(duration, log=['ikr.IKr'], log_interval=25)
    except myokit.SimulationError:
        return float('inf')
    data = data.npview()
    e = np.sum((data['ikr.IKr'] - real['current'])**2)
    return e

# Print score of true solution
print('Score of true solution:')
print(score(real_values))

# Benchmark
print('Benchmark:')
b = myokit.Benchmarker()
n = 1
for i in xrange(n):
    score(real_values)
print(str(b.time() / n) + ' seconds per evaluation')








#
#
# Just for fun: try PSO
#
#
from myokit.lib import fit

# Define a smaller parameter set
parameters = [
    'ikr.p1',
    'ikr.p2',
    ]
    
# And update the score function with these parameters
simulation = myokit.Simulation(model, protocol)
def score(p):
    simulation.reset()
    for i, name in enumerate(parameters):
        simulation.set_constant(name, p[i])
    try:
        data = simulation.run(duration, log=['ikr.IKr'], log_interval=25)
    except myokit.SimulationError:
        return float('inf')
    data = data.npview()
    e = np.sum((data['ikr.IKr'] - real['current'])**2)
    return e

# Set some wide bounds
bounds = [
    (-100, 10000),
    (-100, 100),
    ]

print('Running particle swarm optimisation...')
print('Should take 1-2 minutes on 4 cores')
with np.errstate(all='ignore'): # Tell numpy not to issue warnings
    # Run the optimisation with 4 particles
    x, f = fit.pso(score, bounds, n=4, parallel=True, max_iter=2000)
print('Final score: ' + str(f))
    
# Show solution
print('Current solution:           Real values:')
for k, v in enumerate(x):
    print(myokit.strfloat(v) + '    ' + myokit.strfloat(real_values[k]))
    
# Refine the solution using nelder-mead
x2, f2 = fit.nelder_mead(score, x)
print('Pure PSO solution:          Refined solution:')
for k, v in enumerate(x):
    print(myokit.strfloat(v) + '    ' + myokit.strfloat(x2[k]))

#
# Plot the parameter space near the real coordinates
#
import matplotlib.pyplot as pl
boundaries = [
    [800, 1050],
    [1, 20],
    ]
n = 50
x, fx = fit.map_grid(score, boundaries, n, parallel=True)
fit.loss_surface_colors(x[:,0], x[:,1], fx, markers=None)
pl.title('Score function near the true solution')
fit.loss_surface_colors(x[:,0], x[:,1], np.log(fx), markers=None)
pl.title('Log of score function near the true solution')
pl.show()

