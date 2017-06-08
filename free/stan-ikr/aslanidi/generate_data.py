#!/usr/bin/env python
import matplotlib.pyplot as pl
import numpy as np
import myokit
import myokit.pacing as pacing

# Get model
m = myokit.load_model('aslanidi-2009.mmt')

# Pre-pace at -120 mV

# Create pacing protocol
tpre  = 2000        # Time before step to variable V
tstep = 5000        # Time at variable V
tpost = 3000        # Time after step to variable V
ttotal = tpre + tstep + tpost
vhold = -80         # V before and after step
vres = 10           # Difference in V between steps
v = np.arange(-100, 50 + 10, 10)
# Create block train protocol
p = pacing.steptrain(
        vsteps=v,
        vhold=vhold,
        tpre=tpre,
        tstep=tstep,
        tpost=tpost)

d = [
    'engine.time',
    'membrane.V',
    'ikr.IKr',
    ]

# Run simulation
s = myokit.Simulation(m, p)
d = s.run(p.characteristic_time(), log=d)

# Plot raw data
pl.figure()
pl.subplot(2,1,1)
pl.plot(d.time(), d['membrane.V'])
pl.subplot(2,1,2)
pl.plot(d.time(), d['ikr.IKr'])

# Plot data as overlapping steps
n = len(v)                  # Number of voltages testsed
d2 = d.npview()             # Transform data lists to numpy arrays
d2 = d2.regularize(0.5)     # Resample at dt=0.5ms
d2 = d2.fold(ttotal)        # Cut data into overlapping pieces
pl.figure()
for k in xrange(n):
    pl.subplot(2,1,1)
    pl.plot(d2.time(), d2['membrane.V', k])
    pl.subplot(2,1,2)
    pl.plot(d2.time(), d2['ikr.IKr', k])

# Store raw data
d.save_csv('simple.csv')

pl.show()
