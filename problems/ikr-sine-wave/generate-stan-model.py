#!/usr/bin/env python2
#
# Load Myokit model, write stan model.
#
import os
import sys
import myokit
import myokit.formats.stan

model = 'beattie-2017-ikr'

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
output = 'ikr.IKr'

# Load model
m = os.path.realpath(os.path.join('models', model + '.mmt'))
m = myokit.load_model(m)

# Create stan exporter
e = myokit.formats.stan.StanExporter()

# Export runnable code
e.runnable(model + '-stan', m, parameters=parameters, output=output)
