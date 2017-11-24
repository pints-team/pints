#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import matplotlib.pyplot as pl
import myokit

data_file = os.path.realpath(os.path.join('sine-wave-data', 'sine-wave.csv'))

#
# Uses Myokit/NumPy to recreate the protocol used in Kylie's paper
#

# Load real data
# As created by output file...
d = myokit.DataLog.load_csv(data_file).npview()
t = d['time']
vreal = d['voltage']
del(d)

# Show real data
pl.figure()
pl.subplot(3,1,1)
pl.plot(t, vreal, label='real')

# Create step data
p = myokit.Protocol()
p.add_step(-80, 250)
p.add_step(-120, 50)
p.add_step(-80, 200)
p.add_step(40, 1000)
p.add_step(-120, 500)
p.add_step(-80, 1000)
p.add_step(-30, 3500)
p.add_step(-120, 500)
p.add_step(-80, 1000)

vsteps = p.create_log_for_times(t).npview()['pace']
pl.subplot(3,1,1)
pl.plot(t, vsteps, label='steps')
pl.legend()

pl.subplot(3,1,2)
pl.plot(t, vreal-vsteps)

pl.subplot(3,1,3)
pl.ylim(-0.1, 1.1)
pl.plot(t, vreal == vsteps)

# Create sine data
vsine  = 54 * np.sin(0.007 * (t - 2500))
vsine += 26 * np.sin(0.037 * (t - 2500))
vsine += 10 * np.sin(0.190 * (t - 2500))
vsine[:30000] = 0
vsine[65000:] = 0

pl.subplot(3,1,1)
pl.plot(t, vsine, label='sine')
pl.legend()

vfake = vsteps + vsine
pl.subplot(3,1,1)
pl.plot(t, vfake, label='fake')
pl.legend()

pl.subplot(3,1,2)
pl.plot(t, vreal-vfake)

pl.subplot(3,1,3)
pl.plot(t, vreal == vfake)

# Done!
pl.show()
