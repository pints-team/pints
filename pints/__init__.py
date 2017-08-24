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
VERSION_INT = 0,0,1
VERSION = '.'.join([str(x) for x in VERSION_INT]); del(x)
FLOAT_FORMAT = '{:< 1.17e}'

#
# Expose pints version
#
def version(formatted=False):
    if formatted:
        return 'Pints ' + VERSION
    else:
        return VERSION_INT

#
# Core classes and methods
#
from _core import ForwardModel, SingleSeriesProblem
from _core import vector

#
# Utility classes and methods
#
from _util import strfloat

#
# Boundaries and prior distributions
#
from _boundaries import Boundaries
from _prior import Prior, ComposedPrior
from _prior import UniformPrior

#
# Log-likelihoods
#
from _log_likelihood import LogLikelihood, BayesianLogLikelihood
from _log_likelihood import GaussianLogLikelihood

#
# Scoring functions
#
from _score import ErrorMeasure, LogLikelihoodBasedError
from _score import SumOfSquaresError, RMSError

#
# Parallel function evaluation
#
from _evaluation import Evaluator, SequentialEvaluator, ParallelEvaluator
from _evaluation import evaluate

#
# Optimisation
#
from _optimisers import Optimiser
from _optimisers import TriangleWaveTransform, InfBoundaryTransform
from _optimisers._cmaes import CMAES, cmaes
from _optimisers._pso import PSO, pso
from _optimisers._snes import SNES, snes
from _optimisers._xnes import XNES, xnes

#
# MCMC
#
from _mcmc import MCMC
from _mcmc._adaptive import AdaptiveCovarianceMCMC, adaptive_covariance_mcmc

#
#TODO
#
#from _mcmc import hierarchical_gibbs_sampler
#from _plot import scatter_grid,plot_trace,scatter_diagonal

