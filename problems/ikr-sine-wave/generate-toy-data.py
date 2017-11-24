#!/usr/bin/env python
import matplotlib.pyplot as pl
import numpy as np
import os
import sys
import myokit
import myokit.pacing as pacing

model_dir = os.path.abspath('models')
write_dir = os.path.abspath('toy-data')

# Fix for consistent data?
if True:
    np.random.seed(5)

def generate_data(model, debug=False):
    """
    Generates some test data for one or all models.
    """
    # Get model
    m = myokit.load_model(os.path.join(model_dir, model + '.mmt'))
    
    # Hold at low potential for a while
    vhold = -80
    p = pacing.constant(vhold)
    s = myokit.Simulation(m, p)
    s.run(10000, log=myokit.LOG_NONE)
    m.set_state(s.state())

    # Set up pacing protocol
    tpre  = 2000        # Time before step to variable V
    tstep = 5000        # Time at variable V
    tpost = 3000        # Time after step to variable V
    ttotal = tpre + tstep + tpost
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

    # Set up logging
    d = [
        'engine.time',
        'membrane.V',
        'ikr.IKr',
        ]

    # Run simulation
    s = myokit.Simulation(m, p)
    d = s.run(p.characteristic_time(), log=d, log_interval=25)
    
    # Convert logged data to numpy arrays
    d = d.npview()
    
    # Plot raw data
    if debug:
        pl.figure()
        pl.subplot(2,1,1)
        pl.plot(d.time(), d['membrane.V'])
        pl.subplot(2,1,2)
        pl.plot(d.time(), d['ikr.IKr'])

    # Plot data as overlapping steps
    if debug:
        d2 = d.fold(ttotal)
        pl.figure()
        for k in xrange(len(v)):
            pl.subplot(2,1,1)
            pl.plot(d2.time(), d2['membrane.V', k])
            pl.subplot(2,1,2)
            pl.plot(d2.time(), d2['ikr.IKr', k])
        del(d2)

    # Rename log entries
    d2 = myokit.DataLog()
    d2['current'] = d['ikr.IKr']
    d2['time'] = d['engine.time']
    d2['voltage'] = d['membrane.V']
    d2.set_time_key('time')
    d = d2
    del(d2)

    # Store raw data
    d.save_csv(os.path.join(write_dir, model + '-no-noise.csv'))
    
    # Add noise
    d['current'] += np.random.normal(0,
        0.01 * np.max(d['current']),
        d['current'].shape)
    
    # Plot noisy data
    if debug:
        pl.figure()
        pl.subplot(2,1,1)
        pl.plot(d.time(), d['voltage'])
        pl.subplot(2,1,2)
        pl.plot(d.time(), d['current'])
    
    # Store noisy data
    d.save_csv(os.path.join(write_dir, model + '-with-noise.csv'))

    if debug:
        pl.show()

# Test single model file
if False:
    model = 'aslanidi-2009-ikr'
    generate_data(model, True)
    import sys
    sys.exit(1)

# Generate data for all model files
models = [
    'aslanidi-2009-ikr',
    'clancy-2001-ikr',
    'courtemanche-1998-ikr',
    'diveroli-2013a-ikr',
    'diveroli-2013b-ikr',
    'fink-2008-ikr',
    'fox-2002-ikr',
    'grandi-2010-ikr',
    'hund-2004-ikr',
    'inada-2009-ikr',
    
    
]

print('Generating test data for all models')
for model in models:
    print(model)
    generate_data(model)
print('Done')
