#
# Root of the pints module.
# Provides access to all shared functionality (optimisation, mcmc, etc.).
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import sys

#
# Version info: Remember to keep this in sync with setup.py!
#
VERSION_INT = 0, 0, 1
VERSION = '.'.join([str(x) for x in VERSION_INT])
if sys.version_info[0] < 3:
    del(x)  # Before Python3, list comprehension iterators leaked

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
# Float format: a float can be converted to a 17 digit decimal and back without
# loss of information
FLOAT_FORMAT = '{: .17e}'

#
# Core classes
#
from ._core import ForwardModel, ForwardModelS1
from ._core import ToyModel
from ._core import TunableMethod
from ._core import SingleOutputProblem, MultiOutputProblem

#
# Utility classes and methods
#
from ._util import strfloat, vector, matrix2d
from ._util import Timer
from ._logger import Logger, Loggable

#
# Logs of probability density functions (not necessarily normalised)
#
from ._log_pdfs import (
    LogPDF,
    LogPrior,
    LogLikelihood,
    LogPosterior,
    ProblemLogLikelihood,
)

#
# Log-priors
#
from ._log_priors import (
    ComposedLogPrior,
    MultivariateNormalLogPrior,
    NormalLogPrior,
    UniformLogPrior,
    StudentTLogPrior,
    CauchyLogPrior,
    HalfCauchyLogPrior,
)

#
# Log-likelihoods
#
from ._log_likelihoods import (
    KnownNoiseLogLikelihood,
    UnknownNoiseLogLikelihood,
    ScaledLogLikelihood,
    StudentTLogLikelihood,
    CauchyLogLikelihood,
    SumOfIndependentLogLikelihoods,
)

#
# Boundaries
#
from ._boundaries import (
    Boundaries,
    LogPDFBoundaries,
    RectangularBoundaries,
)

#
# Error measures
#
from ._error_measures import (
    ErrorMeasure,
    ProblemErrorMeasure,
    ProbabilityBasedError,
    SumOfErrors,
    MeanSquaredError,
    RootMeanSquaredError,
    SumOfSquaresError,
)

#
# Parallel function evaluation
#
from ._evaluation import (
    evaluate,
    Evaluator,
    ParallelEvaluator,
    SequentialEvaluator,
)


#
# Optimisation
#
from ._optimisers import (
    Optimiser,
    PopulationBasedOptimiser,
    TriangleWaveTransform,
    Optimisation,
    optimise,
    fmin,
    curve_fit,
)
from ._optimisers._cmaes import CMAES
from ._optimisers._pso import PSO
from ._optimisers._snes import SNES
from ._optimisers._xnes import XNES


#
# Diagnostics
#
from ._diagnostics import (
    effective_sample_size,
    rhat,
    rhat_all_params,
)


#
#  MCMC
#
from ._mcmc import (
    MCMCSampler,
    SingleChainMCMC,
    MultiChainMCMC,
    MCMCSampling,
    mcmc_sample,
)
from ._mcmc._adaptive_covariance import AdaptiveCovarianceMCMC
from ._mcmc._adaptive_covariance_remi import AdaptiveCovarianceRemiMCMC
from ._mcmc._adaptive_covariance_am import AdaptiveCovarianceAMMCMC
from ._mcmc._adaptive_covariance_rao_blackwell import AdaptiveCovarianceRaoBlackWellMCMC
from ._mcmc._adaptive_covariance_global_adaptive import AdaptiveCovarianceGlobalAdaptiveMCMC
from ._mcmc._metropolis import MetropolisRandomWalkMCMC
from ._mcmc._differential_evolution import DifferentialEvolutionMCMC
from ._mcmc._population import PopulationMCMC
from ._mcmc._dream import DreamMCMC


#
# Nested samplers
#
from ._nested import NestedSampler
from ._nested._rejection import NestedRejectionSampler
from ._nested._ellipsoid import NestedEllipsoidSampler


#
# Noise generators (always import!)
#
import pints.noise

#
# Remove any imported modules, so we don't expose them as part of pints
#
del(sys)
