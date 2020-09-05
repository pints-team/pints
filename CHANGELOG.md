# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- This CHANGELOG file to show the changes introduced in each release [#1204](https://github.com/pints-team/pints/pull/1204).
- A new `ConstantAndMultiplicativeGaussianLogLikelihood` was added [#1190](https://github.com/pints-team/pints/pull/1190).
- A new `NoUTurnMCMC` sampler (NUTS) was added, along with a `DualAveragingAdaption` class to adaptively tune related Hamiltonian Monte Carlo methods [#1112](https://github.com/pints-team/pints/pull/1112).
- Three new methods were added for diagnosing autocorrelated noise: `plot_residuals_binned_autocorrelation`, `plot_residuals_binned_std`, and `plot_residuals_distance` [#1183](https://github.com/pints-team/pints/pull/1183).
- A new `Transformation` abstract class was added, along with `ComposedTransformation`, `IdentityTransformation`, `LogitTransformation`, `LogTransformation`, `RectangularBoundariesTransformation`, `ScalingTransformation` subclasses to achieve more effective and efficient optimisation and sampling [#1165](https://github.com/pints-team/pints/pull/1165).
- A new optional argument `transform` was added to both `OptimisationController` and `MCMCController` to transform parameters during optimisation and sampling [#1165](https://github.com/pints-team/pints/pull/1165).
### Changed
- Warnings are now emitted using `warnings.warn` rather than `logging.getLogger(..).warning`. This makes them show up like other warnings, and allows them to be suppressed with [filterwarnings](https://docs.python.org/3/library/warnings.html#warnings.filterwarnings) [#1191](https://github.com/pints-team/pints/pull/1191).
- The new NUTS method is only supported on Python 3.3 and newer; a warning will be emitted when importing PINTS in older versions [#1112](https://github.com/pints-team/pints/pull/1112).
- The installation instructions have been updated to reflect that PINTS in now pip-installable [#1195](https://github.com/pints-team/pints/pull/1195).
- The `pints.Logger` can now deal with `None` being logged in place of a proper value [#1112](https://github.com/pints-team/pints/pull/1112).
### Deprecated
- The method `pints.rhat_all_params` was accidentally removed in 0.3.0, but is now back in deprecated form [#1201](https://github.com/pints-team/pints/pull/1201).
### Fixed
- The output of the method `pints.HalfCauchyLogPrior.sample` had the wrong shape [#1196](https://github.com/pints-team/pints/pull/1196).


## [0.3.0] - 2020-08-08
- This is the first pip installable release. The changelog starts documents all changes since this release.

