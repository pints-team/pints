#
# Root of the pints module.
# Provides access to all shared functionality (optimisation, mcmc, etc.).
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
"""
Pints: Probabilistic Inference on Noisy Time Series.

This module provides several optimisation and sampling methods that can be
applied to find the parameters of a model (typically a time series model) that
are most likely, given an experimental data set.
"""
import os
import sys

#
# Check Python version
#
if sys.hexversion < 0x03050000:  # pragma: no cover
    raise RuntimeError('PINTS requires Python 3.5 or newer.')


#
# Version info
#
def _load_version_int():
    try:
        root = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(root, 'version'), 'r') as f:
            version = f.read().strip().split(',')
        major, minor, revision = [int(x) for x in version]
        return major, minor, revision
    except Exception as e:      # pragma: no cover
        raise RuntimeError('Unable to read version number (' + str(e) + ').')


__version_int__ = _load_version_int()
__version__ = '.'.join([str(x) for x in __version_int__])


#
# Expose pints version
#
def version(formatted=False):
    """
    Returns the version number, as a 3-part integer (major, minor, revision).
    If ``formatted=True``, it returns a string formatted version (for example
    "Pints 1.0.0").
    """
    if formatted:
        return 'Pints ' + __version__
    else:
        return __version_int__


#
# Constants
#
# Float format: a float can be converted to a 17 digit decimal and back without
# loss of information
FLOAT_FORMAT = '{: .17e}'

#
# Core classes
#
from ._core import ForwardModel, ForwardModelS1  # noqa
from ._core import TunableMethod  # noqa
from ._core import SingleOutputProblem, MultiOutputProblem  # noqa

#
# Utility classes and methods
#
from ._util import strfloat, vector, matrix2d  # noqa
from ._util import Timer  # noqa
from ._logger import Logger, Loggable  # noqa

#
# Logs of probability density functions (not necessarily normalised)
#
from ._log_pdfs import (  # noqa
    LogPDF,
    LogPrior,
    LogPosterior,
    PooledLogPDF,
    ProblemLogLikelihood,
    SumOfIndependentLogPDFs,
)

#
# Log-priors
#
from ._log_priors import (  # noqa
    BetaLogPrior,
    BinomialLogPrior,
    CauchyLogPrior,
    ComposedLogPrior,
    ExponentialLogPrior,
    GammaLogPrior,
    GaussianLogPrior,
    HalfCauchyLogPrior,
    InverseGammaLogPrior,
    LogNormalLogPrior,
    MultivariateGaussianLogPrior,
    NegBinomialLogPrior,
    NormalLogPrior,
    StudentTLogPrior,
    TruncatedGaussianLogPrior,
    UniformLogPrior,
)

#
# Log-likelihoods
#
from ._log_likelihoods import (  # noqa
    AR1LogLikelihood,
    ARMA11LogLikelihood,
    CauchyLogLikelihood,
    ConstantAndMultiplicativeGaussianLogLikelihood,
    GaussianIntegratedUniformLogLikelihood,
    GaussianKnownSigmaLogLikelihood,
    GaussianLogLikelihood,
    KnownNoiseLogLikelihood,
    LogNormalLogLikelihood,
    MultiplicativeGaussianLogLikelihood,
    ScaledLogLikelihood,
    StudentTLogLikelihood,
    UnknownNoiseLogLikelihood,
)

#
# Boundaries
#
from ._boundaries import (  # noqa
    Boundaries,
    LogPDFBoundaries,
    RectangularBoundaries,
)

#
# Error measures
#
from ._error_measures import (  # noqa
    ErrorMeasure,
    MeanSquaredError,
    NormalisedRootMeanSquaredError,
    ProbabilityBasedError,
    ProblemErrorMeasure,
    RootMeanSquaredError,
    SumOfErrors,
    SumOfSquaresError,
)

#
# Parallel function evaluation
#
from ._evaluation import (  # noqa
    evaluate,
    Evaluator,
    ParallelEvaluator,
    SequentialEvaluator,
    MultiSequentialEvaluator,
)


#
# Optimisation
#
from ._optimisers import (  # noqa
    curve_fit,
    fmin,
    Optimisation,
    OptimisationController,
    optimise,
    Optimiser,
    PopulationBasedOptimiser,
)

from ._optimisers._adam import Adam
from ._optimisers._cmaes import CMAES
from ._optimisers._cmaes_bare import BareCMAES
from ._optimisers._gradient_descent import GradientDescent
from ._optimisers._irpropmin import IRPropMin
from ._optimisers._nelder_mead import NelderMead
from ._optimisers._pso import PSO
from ._optimisers._snes import SNES
from ._optimisers._xnes import XNES


#
# Diagnostics
#
from ._diagnostics import (  # noqa
    effective_sample_size,
    rhat,
    rhat_all_params,
)


#
#  MCMC
#
from ._mcmc import (  # noqa
    mcmc_sample,
    MCMCController,
    MCMCSampler,
    MCMCSampling,
    MultiChainMCMC,
    SingleChainMCMC,
)
# base classes first
from ._mcmc._adaptive_covariance import AdaptiveCovarianceMC  # noqa

# methods
from ._mcmc._differential_evolution import DifferentialEvolutionMCMC  # noqa
from ._mcmc._dram_ac import DramACMC  # noqa
from ._mcmc._dream import DreamMCMC  # noqa
from ._mcmc._dual_averaging import DualAveragingAdaption  # noqa
from ._mcmc._emcee_hammer import EmceeHammerMCMC  # noqa
from ._mcmc._haario_ac import HaarioACMC  # noqa
from ._mcmc._haario_bardenet_ac import HaarioBardenetACMC  # noqa
from ._mcmc._haario_bardenet_ac import AdaptiveCovarianceMCMC  # noqa
from ._mcmc._hamiltonian import HamiltonianMCMC  # noqa
from ._mcmc._mala import MALAMCMC  # noqa
from ._mcmc._metropolis import MetropolisRandomWalkMCMC  # noqa
from ._mcmc._monomial_gamma_hamiltonian import MonomialGammaHamiltonianMCMC  # noqa
from ._mcmc._nuts import NoUTurnMCMC  # noqa
from ._mcmc._population import PopulationMCMC  # noqa
from ._mcmc._rao_blackwell_ac import RaoBlackwellACMC  # noqa
from ._mcmc._relativistic import RelativisticMCMC  # noqa
from ._mcmc._slice_doubling import SliceDoublingMCMC  # noqa
from ._mcmc._slice_rank_shrinking import SliceRankShrinkingMCMC  # noqa
from ._mcmc._slice_stepout import SliceStepoutMCMC  # noqa
from ._mcmc._summary import MCMCSummary  # noqa


#
# Nested samplers
#
from ._nested import NestedSampler  # noqa
from ._nested import NestedController  # noqa
from ._nested._rejection import NestedRejectionSampler  # noqa
from ._nested._ellipsoid import NestedEllipsoidSampler  # noqa


#
# ABC
#
from ._abc import ABCSampler
from ._abc import ABCController
from ._abc._abc_rejection import RejectionABC
from ._abc._abc_smc import ABCSMC


#
# Sampling initialising
#
from ._sample_initial_points import sample_initial_points  # noqa


#
# Transformations
#
from ._transformation import (  # noqa
    ComposedTransformation,
    IdentityTransformation,
    LogitTransformation,
    LogTransformation,
    RectangularBoundariesTransformation,
    ScalingTransformation,
    Transformation,
    TransformedBoundaries,
    TransformedErrorMeasure,
    TransformedLogPDF,
    TransformedLogPrior,
    TransformedRectangularBoundaries,
    UnitCubeTransformation,
)


#
# Noise generators (always import!)
#
from . import noise  # noqa

#
# Remove any imported modules, so we don't expose them as part of pints
#
del os, sys
