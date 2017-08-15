#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as pl
import pystan
import myokit

# Load data
d = myokit.DataLog.load_csv('simple.csv')
t = d['engine.time']
V = d['membrane.V']
I = d['ikr.IKr']

# Show data
if False:
    pl.figure()
    pl.subplot(2,1,1)
    pl.plot(t, V)
    pl.subplot(2,1,2)
    pl.plot(t, I)
    pl.show()

# Show stan file
model = pystan.StanModel('aslanidi-2009.stan')
