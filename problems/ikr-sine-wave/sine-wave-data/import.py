#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import numpy as np
import scipy
import scipy.io
import matplotlib.pyplot as pl
import myokit
import myokit.formats.axon

show_debug = True

cell = 'cell-5'

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

# Load protocol from protocol file
mat = 'sine_wave_protocol.mat'
mat = scipy.io.loadmat(mat)
vm = mat['T']
vm = vm[:,0]  # Convert from matrix to array
del(mat)

# Load leak-corrected, dofetilide-subtracted IKr data from matlab file
mat = 'sine_wave_' + idx + '_dofetilide_subtracted_leak_subtracted.mat'
mat = scipy.io.loadmat(mat)
current = mat['T']
current = current[:,0]  # Convert from matrix to array
del(mat)

# Create times array, using dt=0.1ms
time = np.arange(len(current)) * 0.1

# Correct tiny shift in stored data (doubling final point)
vm[:-1] = vm[1:]
current[:-1] = current[1:]

# Show data with capacitance artefacts
if show_debug:
    pl.figure()
    pl.subplot(4,1,1)
    pl.plot(time, vm)
    pl.subplot(4,1,2)
    pl.plot(time, current)

# Remove capacitance artefacts
cap_duration = 1.5
jumps = [
    250,
    300,
    500,
    1500,
    2000,
    3000,
    6500,
    7000,    
    ]
for t in jumps:
    # Get indices of capacitance start and end
    i1 = (np.abs(time-t)).argmin()
    i2 = (np.abs(time-t-cap_duration)).argmin()
    # Remove data points during capacitance artefact
    #time = np.concatenate((time[:i1], time[i2:]))
    #current = np.concatenate((current[:i1], current[i2:]))
    #vm = np.concatenate((vm[:i1], vm[i2:]))
    current[i1:i2] = np.mean(current[i1-(i2-i1): i1])

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
d.save_csv(cell + '.csv')


# Show debug data
if show_debug:
    pl.show()
