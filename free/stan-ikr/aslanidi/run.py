#!/usr/bin/env python
import numpy as np
import myokit
import matplotlib.pyplot as pl
import pystan

d = myokit.DataLog.load_csv('simple.csv').npview()

t = d['engine.time']
V = d['membrane.V']
I = d['ikr.IKr']

if False:
    pl.figure()
    pl.subplot(2,1,1)
    pl.plot(t, V)
    pl.subplot(2,1,2)
    pl.plot(t, I)
    pl.show()

model = pystan.StanModel('aslanidi-2009.stan')
