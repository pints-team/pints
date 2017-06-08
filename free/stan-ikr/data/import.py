#!/usr/bin/env python
from __future__ import print_function
import myokit
import myokit.formats.axon
import numpy as np
import scipy
import scipy.io
import matplotlib.pyplot as pl

show_debug = True

cell = 'cell-1'

cells = {
    'cell-1': '16713003',
    'cell-2': '16715049',
    'cell-3': '16708016',
    'cell-4': '16708060',
    'cell-5': '16713110',
    'cell-6': '16708118',
    'cell-7': '16704007',
    'cell-8': '16704047',
    'cell-9': '16707014',
}

idx = cells[cell]
abf = 'raw/' + idx + '/sine_wave.abf'
mat = 'sine_wave_' + idx + '_dofetilide_subtracted_leak_subtracted.mat'

# Load times and voltage from abf file
abf = myokit.formats.axon.AbfFile(abf)
abf = abf.myokit_log().npview()
time = abf.time()
vm = abf['1.ad']
del(abf)

# Load leak-corrected, dofetilide-subtracted IKr data from matlab file
mat = scipy.io.loadmat(mat)
current = mat['T']
current = current[:,0]  # Convert from matrix to array
del(mat)

# Show data with capacitance artefacts
if show_debug:
    pl.figure()
    pl.subplot(4,1,1)
    pl.plot(time, vm)
    pl.subplot(4,1,2)
    pl.plot(time, current)

# Remove capacitance artefacts
cap_duration = 0.0015
jumps = [
    0.25,
    0.3,
    0.5,
    1.5,
    2,
    3,
    6.5,
    7,    
    ]
for t in jumps:
    # Get indices of capacitance start and end
    i1 = (np.abs(time-t)).argmin()
    i2 = (np.abs(time-t-cap_duration)).argmin()
    # Remove data points during capacitance artefact
    time = np.concatenate((time[:i1], time[i2:]))
    current = np.concatenate((current[:i1], current[i2:]))
    vm = np.concatenate((vm[:i1], vm[i2:]))

# Show data without capacitance artefacts
if show_debug:
    pl.subplot(4,1,3)
    pl.plot(time, vm)
    pl.subplot(4,1,4)
    pl.plot(time, current)

# Store in csv
d = myokit.DataLog()
d['time'] = time
d['voltage'] = vm
d['current'] = current
d.save_csv('real-data.csv')


# Show debug data
if show_debug:
    pl.show()
