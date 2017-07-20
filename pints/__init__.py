#
# Root of the pints module.
# Provides access to all shared functionality (optimisation, mcmc, etc.).
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
# Core classes
from _forward_model import ForwardModel
#
from _plot import scatter_grid,plot_trace,scatter_diagonal
#
from _prior import Prior,Uniform,Normal
#
import optimise
import mcmc
