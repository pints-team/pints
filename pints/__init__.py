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
# Version info: Remember to keep this in sync with setup.py!
#
VERSION_INT = 0,0,1
VERSION = '.'.join([str(x) for x in VERSION_INT]); del(x)

#
# Expose pints version
#
def version(formatted=False):
    if formatted:
        return 'Pints ' + VERSION
    else:
        return VERSION_INT

#
# Constants
#
FLOAT_FORMAT = '{:< 1.17e}'

#
# Core classes
#
from _core import ForwardModel, SingleSeriesProblem

#
# Utility classes and methods
#
from _util import strfloat, vector

#
# Boundaries and prior distributions
#
from _boundaries import Boundaries
from _prior import Prior, ComposedPrior
from _prior import UniformPrior, NormalPrior, MultivariateNormalPrior

#
# Log-likelihoods
#
from _log_likelihood import LogLikelihood, LogPosterior
from _log_likelihood import KnownNoiseLogLikelihood, UnknownNoiseLogLikelihood
from _log_likelihood import ScaledLogLikelihood

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

