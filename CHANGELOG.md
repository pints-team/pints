# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
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
- [#1250](https://github.com/pints-team/pints/pull/1250) The returned values from `SingleChainMCMC.tell()` and `MultiChainMCMC.tell()` have been extended from current position `x` to `x, fx, accepted`, where `fx` is the current log likelihood and `accepted` is a bool indicating whether tell performed an acceptance step in this call.
- [#1195](https://github.com/pints-team/pints/pull/1195) The installation instructions have been updated to reflect that PINTS in now pip-installable.
- [#1191](https://github.com/pints-team/pints/pull/1191) Warnings are now emitted using `warnings.warn` rather than `logging.getLogger(..).warning`. This makes them show up like other warnings, and allows them to be suppressed with [filterwarnings](https://docs.python.org/3/library/warnings.html#warnings.filterwarnings).
- [#1112](https://github.com/pints-team/pints/pull/1112) The `pints.Logger` can now deal with `None` being logged in place of a proper value.
- [#1355](https://github.com/pints-team/pints/pull/1355) When called with `parallel=True` the method `pints.evaluate()` will now limit the number of workers it uses to the number of tasks it needs to process.
- [#1357](https://github.com/pints-team/pints/pull/1357) Parallel evaluations using multiprocessing now restrict the number of threads used by Numpy and others to 1 (by default).
- [#1365](https://github.com/pints-team/pints/pull/1365) Dropped support for Python 2.7. PINTS now requires Python 3.5 or higher.
### Deprecated
- [#1201](https://github.com/pints-team/pints/pull/1201) The method `pints.rhat_all_params` was accidentally removed in 0.3.0, but is now back in deprecated form.
### Removed
- [#1250](https://github.com/pints-team/pints/pull/1250) The methods `SingleChainMCMC.current_log_pdf()` and `MultiChainMCMC.current_log_pdf()` have been removed.
### Fixed
- [#1264](https://github.com/pints-team/pints/pull/1264) Fixed a bug relating to how NUTS handles nans when values outside the range of the priors are proposed.
- [#1257](https://github.com/pints-team/pints/pull/1257) Fixed a bug in `GaussianLogPrior`, which meant the distribution could be instantiated with a non-positive standard deviation.
- [#1246](https://github.com/pints-team/pints/pull/1246) Fixed a long-standing bug in `PopulationMCMC`, which caused it to sample incorrectly.
- [#1196](https://github.com/pints-team/pints/pull/1196) The output of the method `pints.HalfCauchyLogPrior.sample` had the wrong shape.


## [0.3.0] - 2020-08-08
- This is the first pip installable release. The changelog documents all changes since this release.
