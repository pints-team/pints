#
# Root of the pints module.
# Provides access to all shared functionality (optimisation, mcmc, etc.).
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
# Optmisation
from _cmaes import fit_model_with_cmaes
# Bayesian inference
from _mcmc import mcmc_with_adaptive_covariance,hierarchical_gibbs_sampler
#
from _plot import scatter_grid,plot_trace,scatter_diagonal
#
from _prior import Prior,Uniform,Normal

