# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added
### Changed
### Deprecated
### Removed
### Fixed


## [0.5.0] - 2023-07-27

### Added
- [#1484](https://github.com/pints-team/pints/pull/1484) Added a `GaussianIntegratedLogUniformLogLikelihood` class that calculates the log-likelihood with its Gaussian noise integrated out with an uninformative prior in log-space.
- [#1466](https://github.com/pints-team/pints/pull/1466) Added a `TransformedRectangularBoundaries` class that preserves the `RectangularBoundaries` methods after transformation.
- [#1462](https://github.com/pints-team/pints/pull/1461) The `OptimisationController` now has a stopping criterion `max_evaluations`.
- [#1460](https://github.com/pints-team/pints/pull/1460) [#1468](https://github.com/pints-team/pints/pull/1468) Added the `Adam` local optimiser.
- [#1459](https://github.com/pints-team/pints/pull/1459) [#1465](https://github.com/pints-team/pints/pull/1465) Added the `iRprop-` local optimiser.
- [#1456](https://github.com/pints-team/pints/pull/1456) Added an optional `translation` to `ScalingTransform` and added a `UnitCubeTransformation` class.
- [#1432](https://github.com/pints-team/pints/pull/1432) Added 2 new stochastic models: production and degradation model, Schlogl's system of chemical reactions. Moved the stochastic logistic model into `pints.stochastic` to take advantage of the `MarkovJumpModel`.
- [#1420](https://github.com/pints-team/pints/pull/1420) The `Optimiser` class now distinguishes between a best-visited point (`x_best`, with score `f_best`) and a best-guessed point (`x_guessed`, with approximate score `f_guessed`). For most optimisers, the two values are equivalent. The `OptimisationController` still tracks `x_best` and `f_best` by default, but this can be modified using the methods `set_f_guessed_tracking` and `f_guessed_tracking`.
- [#1417](https://github.com/pints-team/pints/pull/1417) Added a module `toy.stochastic` for stochastic models. In particular, `toy.stochastic.MarkovJumpModel` implements Gillespie's algorithm for easier future implementation of stochastic models.
- [#1413](https://github.com/pints-team/pints/pull/1413) Added classes `pints.ABCController` and `pints.ABCSampler` for Approximate Bayesian computation (ABC) samplers. Added `pints.RejectionABC` which implements a simple rejection ABC sampling algorithm.
- [#1378](https://github.com/pints-team/pints/pull/1378) Added a class `pints.LogNormalLogLikelihood`.

### Changed
- [#1485](https://github.com/pints-team/pints/pull/1485) PINTS is no longer tested on Ubuntu 18.04 LTS, but on 20.04 LTS and 22.04 LTS.
- [#1479](https://github.com/pints-team/pints/pull/1479) PINTS is no longer tested on Python 3.6. Testing for Python 3.11 has been added.
- [#1479](https://github.com/pints-team/pints/pull/1479) The `asyncio.coroutine` decorators have been removed from all of NUTS's coroutines in order to be compatible with Python 3.11.
- [#1466](https://github.com/pints-team/pints/pull/1466) `Transformation.convert_boundaries` will now return a `TransformedRectangularBoundaries` object if the transformation is element-wise and the given boundaries extend `RectangularBoundaries`.
- [#1458](https://github.com/pints-team/pints/pull/1458) The `GradientDescent` optimiser now sets its default learning rate as `min(sigma0)` (it can still be changed afterwards with `set_learning_rate()`).
- [#1445](https://github.com/pints-team/pints/pull/1445) Allowed multiple LogPDFs to be supplied to the MCMCController (one for each MCMC chain), and added an evaluator which evaluates each position on a separate callable.
- [#1439](https://github.com/pints-team/pints/pull/1439), [#1433](https://github.com/pints-team/pints/pull/1433) PINTS is no longer tested on Python 3.5. Testing for Python 3.10 has been added.
- [#1435](https://github.com/pints-team/pints/pull/1435) The optional Stan interface now uses (and requires) pystan 3 or newer. The ``update_data`` method has been remove (model compilation is now cached so that there is no performance benefit to using this method).
- [#1424](https://github.com/pints-team/pints/pull/1424) Fixed a bug in PSO that caused it to use more particles than advertised.
- [#1424](https://github.com/pints-team/pints/pull/1424) xNES, SNES, PSO, and BareCMAES no longer use a `TriangleWaveTransform` to handle rectangular boundaries (this was found to lead to optimisers diverging in some cases).

### Removed
- [#1424](https://github.com/pints-team/pints/pull/1424) Removed the `TriangleWaveTransform` class previously used in some optimisers.

### Fixed
- [#1457](https://github.com/pints-team/pints/pull/1457) Fixed typo in deprecation warning for `UnknownNoiseLikelihood`.
- [#1455](https://github.com/pints-team/pints/pull/1455) The `s` and `inv_s` properties of `ScalingTransformation` have been replaced with private properties `_s` and `_inv_s`.
- [#1450](https://github.com/pints-team/pints/pull/1450) Made `TransformedBoundaries` consistent with `Boundaries` by removing `range()` and adding `sample()`.
- [#1449](https://github.com/pints-team/pints/pull/1449) Fixed a bug in `MarkovJumpModel.interpolate_mol_counts`.
- [#1399](https://github.com/pints-team/pints/pull/1339) Fixed a bug in `DramACMC`, and fixed the number of proposal kernels to 2.


## [0.4.0] - 2021-12-07

### Added
- [#1409](https://github.com/pints-team/pints/pull/1409) The `OptimisationController` now accepts a callback function that will be called at every iteration; this can be used for easier customisation or visualisation of the optimiser trajectory.
- [#1383](https://github.com/pints-team/pints/pull/1383) Added a method `toy.TwistedGaussianDistribution.untwist` that turns samples from this distribution into samples from a multivariate Gaussian.
- [#1322](https://github.com/pints-team/pints/pull/1322) Added a method `sample_initial_points` that allows users to generate random points with finite metrics (either log-probabilities or error measures) to use as starting points for sampling or optimisation.
- [#1243](https://github.com/pints-team/pints/pull/1243) Added testing for Python 3.9.
- [#1213](https://github.com/pints-team/pints/pull/1213), [#1216](https://github.com/pints-team/pints/pull/1216) Added the truncated Gaussian distribution as a log prior, `TruncatedGaussianLogPrior`.
- [#1212](https://github.com/pints-team/pints/pull/1213) Added the `PooledLogPDF` class to allow for pooling parameters across log-pdfs.
- [#1204](https://github.com/pints-team/pints/pull/1204) This CHANGELOG file to show the changes introduced in each release.
- [#1190](https://github.com/pints-team/pints/pull/1190) A new `ConstantAndMultiplicativeGaussianLogLikelihood` was added.
- [#1183](https://github.com/pints-team/pints/pull/1183) Three new methods were added for diagnosing autocorrelated or time-varying noise: `plot_residuals_binned_autocorrelation`, `plot_residuals_binned_std`, and `plot_residuals_distance`.
- [#1175](https://github.com/pints-team/pints/pull/1175) Added notebooks showing how to interface with the `statsmodels` Python package which allows fitting ARIMAX and state space models in PINTS.
- [#1165](https://github.com/pints-team/pints/pull/1165) A new `Transformation` abstract class was added, along with `ComposedTransformation`, `IdentityTransformation`, `LogitTransformation`, `LogTransformation`, `RectangularBoundariesTransformation`, `ScalingTransformation` subclasses to achieve more effective and efficient optimisation and sampling.
- [#1165](https://github.com/pints-team/pints/pull/1165) A new optional argument `transform` was added to both `OptimisationController` and `MCMCController` to transform parameters during optimisation and sampling.
- [#1112](https://github.com/pints-team/pints/pull/1112) A new `NoUTurnMCMC` sampler (NUTS) was added, along with a `DualAveragingAdaption` class to adaptively tune related Hamiltonian Monte Carlo methods.
- [#1025](https://github.com/pints-team/pints/pull/1025) Added a stochastic logistic growth problem for use with ABC.

### Changed
- [#1420](https://github.com/pints-team/pints/pull/1420) The `OptimisationController` now logs a best and a current score.
- [#1375](https://github.com/pints-team/pints/pull/1375) Changed all arguments called `transform` to `transformation` for consistency.
- [#1365](https://github.com/pints-team/pints/pull/1365) Dropped support for Python 2.7. PINTS now requires Python 3.5 or higher.
- [#1360](https://github.com/pints-team/pints/pull/1360) The `ParallelEvaluator` will now set a different (pre-determined) random seed for each task, ensuring tasks can use randomness, but results can be reproduced from run to run.
- [#1357](https://github.com/pints-team/pints/pull/1357) Parallel evaluations using multiprocessing now restrict the number of threads used by Numpy and others to 1 (by default).
- [#1355](https://github.com/pints-team/pints/pull/1355) When called with `parallel=True` the method `pints.evaluate()` will now limit the number of workers it uses to the number of tasks it needs to process.
- [#1250](https://github.com/pints-team/pints/pull/1250) The returned values from `SingleChainMCMC.tell()` and `MultiChainMCMC.tell()` have been extended from current position `x` to `x, fx, accepted`, where `fx` is the current log likelihood and `accepted` is a bool indicating whether tell performed an acceptance step in this call.
- [#1195](https://github.com/pints-team/pints/pull/1195) The installation instructions have been updated to reflect that PINTS in now pip-installable.
- [#1191](https://github.com/pints-team/pints/pull/1191) Warnings are now emitted using `warnings.warn` rather than `logging.getLogger(..).warning`. This makes them show up like other warnings, and allows them to be suppressed with [filterwarnings](https://docs.python.org/3/library/warnings.html#warnings.filterwarnings).
- [#1112](https://github.com/pints-team/pints/pull/1112) The `pints.Logger` can now deal with `None` being logged in place of a proper value.

### Deprecated
- [#1420](https://github.com/pints-team/pints/pull/1420) The methods `pints.Optimisation.xbest()` and `fbest()` are deprecated in favour of `x_best()` and `f_best()`.
- [#1201](https://github.com/pints-team/pints/pull/1201) The method `pints.rhat_all_params` was accidentally removed in 0.3.0, but is now back in deprecated form.

### Removed
- [#1250](https://github.com/pints-team/pints/pull/1250) The methods `SingleChainMCMC.current_log_pdf()` and `MultiChainMCMC.current_log_pdf()` have been removed.

### Fixed
- [#1350](https://github.com/pints-team/pints/pull/1350) Fixed bugs in the Relativistic MCMC sampler.
- [#1264](https://github.com/pints-team/pints/pull/1264) Fixed a bug relating to how NUTS handles nans when values outside the range of the priors are proposed.
- [#1257](https://github.com/pints-team/pints/pull/1257) Fixed a bug in `GaussianLogPrior`, which meant the distribution could be instantiated with a non-positive standard deviation.
- [#1246](https://github.com/pints-team/pints/pull/1246) Fixed a long-standing bug in `PopulationMCMC`, which caused it to sample incorrectly.
- [#1196](https://github.com/pints-team/pints/pull/1196) The output of the method `pints.HalfCauchyLogPrior.sample` had the wrong shape.


## [0.3.0] - 2020-08-08
- This is the first pip installable release. The changelog documents all changes since this release.
