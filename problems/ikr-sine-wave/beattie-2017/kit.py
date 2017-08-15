#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join('..', '..')))
sys.path.append(os.path.abspath(os.path.join('..', 'myokit')))
import myokit
import myokit.pacing as pacing
from myokit.lib import fit

model_file = os.path.realpath(os.path.join('..', 'models', 
    'beattie-2017-ikr.mmt'))
data_file = os.path.realpath(os.path.join('..', 'sine-wave-data',
    'cell-5.csv'))

#
# Fit Kylie's model to real data
# 

# Get optimization method from arguments
methods = ['cmaes', 'pso', 'snes', 'xnes', 'hybrid', 'show']
method = ''
if len(sys.argv) == 2:
    method = sys.argv[1]
    if method not in methods:
        print('Unknown method "' + str(method) + '"')
        print('Choose one of: ' + ', '.join(methods))
        sys.exit(1)

# Load data
real = myokit.DataLog.load_csv(data_file).npview()

# Load model
model = myokit.load_model(model_file)

# Create model for pre-pacing
model_pre = model.clone()

# Create pre-pacing protocol
protocol_pre = myokit.pacing.constant(-80)

# Create step protocol
protocol = myokit.Protocol()
protocol.add_step(-80, 250)
protocol.add_step(-120, 50)
protocol.add_step(-80, 200)
protocol.add_step(40, 1000)
protocol.add_step(-120, 500)
protocol.add_step(-80, 1000)
protocol.add_step(-30, 3500)
protocol.add_step(-120, 500)
protocol.add_step(-80, 1000)
duration = protocol.characteristic_time()

# Create capacitance filter
cap_duration = 1.5
dt = 0.1
fcap = np.ones(int(duration / dt), dtype=int)
for event in protocol:
    i1 = int(event.start() / dt)
    i2 = i1 + int(cap_duration / dt)
    if i1 > 0: # Skip first one
        fcap[i1:i2] = 0

# Change RHS of membrane.V
model.get('membrane.V').set_rhs('if(engine.time < 3000 or engine.time >= 6500,'
    + ' engine.pace, '
    + ' - 30'
    + ' + 54 * sin(0.007 * (engine.time - 2500))'
    + ' + 26 * sin(0.037 * (engine.time - 2500))'
    + ' + 10 * sin(0.190 * (engine.time - 2500))'
    + ')')

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
    'ikr.p9',
    ]
boundaries = {
    'ikr.p1' : [1e-7, 1e-1],     # 1/ms
    'ikr.p2' : [1e-7, 1e-1],     # 1/mV
    'ikr.p3' : [1e-7, 1e-1],     # 1/ms
    'ikr.p4' : [1e-7, 1e-1],     # 1/mV
    'ikr.p5' : [1e-7, 1e-1],     # 1/ms
    'ikr.p6' : [1e-7, 1e-1],     # 1/mV
    'ikr.p7' : [1e-7, 1e-1],     # 1/ms
    'ikr.p8' : [1e-7, 1e-1],     # 1/mV
    'ikr.p9' : [1e-5, 1e3],     # mS/uF
    }

bounds = [boundaries[x] for x in parameters]

# define score function (sum of squares)
simulation = myokit.Simulation(model, protocol)
#simulation.set_tolerance(1e-8, 1e-8)
simulation_pre = myokit.Simulation(model_pre, protocol_pre)
def score(p):
    # Simulate
    for i, name in enumerate(parameters):
        simulation.set_constant(name, p[i])
        simulation_pre.set_constant(name, p[i])
    try:
        simulation_pre.reset()
        simulation_pre.pre(10000)
        simulation.reset()
        simulation.set_state(simulation_pre.state())        
        data = simulation.run(duration, log=['ikr.IKr'], log_interval=0.1)
    except myokit.SimulationError:
        return float('inf')
    # Calculate error and return
    e = fcap * (np.asarray(data['ikr.IKr']) - real['current'])
    return np.sum(e**2)

#hint = np.ones(9) * 1e-3
hint = 'random'
hints = None if hint == 'random' else [hint]

# Run
target = 0
xs, fs = [], []
repeats = 10
for i in xrange(repeats):
    if method == 'pso':
        print('Running PSO')
        with np.errstate(all='ignore'): # Tell numpy not to issue warnings
            x, f = fit.pso(score, bounds, n=96, parallel=True,
                hints=hints, max_iter=1000, verbose=20)
    elif method == 'xnes':
        print('Running xNES')
        with np.errstate(all='ignore'):
          x, f = fit.xnes(score, bounds, parallel=True, target=target,
                hint=hint, verbose=20)
    elif method == 'cmaes':
        print('Running CMA-ES')
        with np.errstate(all='ignore'):
            x, f = fit.cmaes(score, bounds, parallel=True, target=target,
                hint=hint, verbose=True)
    elif method == 'snes':
        print('Running SNES')
        with np.errstate(all='ignore'):
            x, f = fit.snes(score, bounds, parallel=True, target=target,
                hint=hint, verbose=20)
    elif method == 'hybrid':
        nbest = 3
        with np.errstate(all='ignore'):
            print('Running shallow PSO to get ' + str(nbest) + ' best')
            xs2, fs2 = fit.pso(score, bounds, n=96, parallel=True,
                target=100, hints=hints, max_iter=500, verbose=20,
                return_all=True)
            for i in xrange(nbest):
                print('Running CMA-ES from pso point ' + str(1 + i))
                x, f = fit.cmaes(score, bounds, parallel=True,
                    hint=xs2[i], sigma=1e-4, verbose=True)
                xs.append(x)
                fs.append(f)
        break
    elif method == 'show':
        # Get file to show
        solution = 'last-solution-kit.txt'
        if len(sys.argv) > 2:
            solution = sys.argv[2]
        # Read last solution
        with open(solution, 'r') as f:
            print('Showing solution from: ' + solution)
            lines = f.readlines()
            if len(lines) < len(parameters):
                print('Unable to read last solution, expected ' + str(len(parameters))
                    + ' lines, only found ' + str(len(lines)))
                sys.exit(1)
            try:
                solution = [float(x.strip()) for x in lines[:len(parameters)]]
            except ValueError:
                print('Unable to read last solution')
                raise
        # Show parameters    
        print('Parameters:')
        for k, v in enumerate(solution):
            print(myokit.strfloat(v))
        # Show score
        print('Score: ' + str(score(solution)))
        # Simulate
        for i, name in enumerate(parameters):
            simulation_pre.set_constant(name, solution[i])
            simulation.set_constant(name, solution[i])
        simulation_pre.reset()
        simulation.reset()
        simulation_pre.pre(10000)        
        simulation.set_state(simulation_pre.state())        
        d = ['engine.time', 'ikr.IKr', 'membrane.V']
        d = simulation.run(duration, log=d, log_interval=0.1)
        # Show plots
        import matplotlib.pyplot as pl
        pl.figure()
        pl.subplot(2,1,1)
        pl.xlabel('Time [ms]')
        pl.ylabel('V [mV]')
        pl.plot(real['time'], real['voltage'], label='real')
        pl.plot(d.time(), d['membrane.V'], label='sim')
        pl.legend(loc='lower right')
        pl.subplot(2,1,2)
        pl.xlabel('Time [ms]')
        pl.plot(real['time'], real['current'], lw=1, label='real')
        pl.plot(real['time'], d['ikr.IKr'], lw=1, label=model.name())
        pl.plot(real['time'], fcap, label='filter')
        pl.legend(loc='lower right')
        pl.show()
        sys.exit(0)        
    else:
        print('Unknown method: "' + str(method) + '"')
        sys.exit(1)
    xs.append(x)
    fs.append(f)

print('All scores:')
for f in fs:
    print('  ' + str(f))

ibest = np.argmin(fs)
x = xs[ibest]
f = fs[ibest]

# Show final score
print('Final score: ' + str(f))
    
# Show solution
print('Final solution:')
for k, v in enumerate(x):
    print(myokit.strfloat(v))

# Store solution
with open('last-solution-kit.txt', 'w') as f:
    for k, v in enumerate(x):
        f.write(myokit.strfloat(v) + '\n')

if False:
    simulation.reset()
    for k, v in zip(parameters, x):
        simulation.set_constant(k, v)
    data = simulation.run(duration, log=['ikr.IKr'], log_interval=0.1)
    import matplotlib.pyplot as pl
    pl.figure()
    pl.plot(real['time'], real['current'], label='real')
    pl.plot(real['time'], data['ikr.IKr'], label='aslanidi')
    pl.plot(real['time'], fcap, label='filter')
    e = fcap * (np.asarray(data['ikr.IKr']) - real['current'])
    pl.plot(real['time'], e, label='error')
    print(np.sum(e**2))
    pl.legend()
    pl.show()
