#
# Root of the pints module.
# Provides access to all shared functionality (optimisation, mcmc, etc.).
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#

#
# Constants
#
VERSION_INT = 1,25,2
VERSION = '.'.join([str(x) for x in VERSION_INT]); del(x)
FLOAT_FORMAT = '{:< 1.17e}'

#
# Core classes and methods
#
from _core import ForwardModel, Boundaries, SingleSeriesProblem
from _core import vector

#
# Utility classes and methods
#
from _util import strfloat

#
# Scoring functions and likelihoods
#
from _score import MeasureOfFit
from _score import SumOfSquaresError, RMSError

#
# Parallel function evaluation
#
from _evaluation import Evaluator, SequentialEvaluator, ParallelEvaluator
from _evaluation import evaluate

#
# Optimisation
#
from _optimisation import Optimiser
from _optimisation import TriangleWaveTransform, InfBoundaryTransform
from _optimisers._cmaes import CMAES, cmaes
from _optimisers._pso import PSO, pso
from _optimisers._snes import SNES, snes
from _optimisers._xnes import XNES, xnes

# MCMC
# Parameter space exploration
# Visualisation

#
#TODO
#
#from _mcmc import mcmc_with_adaptive_covariance
#from _mcmc import hierarchical_gibbs_sampler
#from _plot import scatter_grid,plot_trace,scatter_diagonal
#from _prior import Prior,Uniform,Normal

