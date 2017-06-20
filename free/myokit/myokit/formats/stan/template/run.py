#!/usr/bin/env python
import pystan

# Load stan file
model = pystan.StanModel('cell.stan')
