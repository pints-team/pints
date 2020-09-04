# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- This CHANGELOG file to show the changes introduced in each release.
- A new `ConstantAndMultiplicativeGaussianLogLikelihood` was added.
- A new `NoUTurnMCMC` sampler (NUTS) was added, along with a `DualAveragingAdaption` class to adaptively tune related Hamiltonian Monte Carlo methods.
- Three new methods were added for diagnosing autocorrelated noise: `plot_residuals_binned_autocorrelation`, `plot_residuals_binned_std`, and `plot_residuals_distance`.
- A new `Transformation` abstract class was added, along with `ComposedTransformation`, `IdentityTransformation`, `LogitTransformation`, `LogTransformation`, `RectangularBoundariesTransformation`, `ScalingTransformation` subclasses to achieve more effective and efficient optimisation and sampling.
### Changed
- Warnings are now emitted using `warnings.warn` rather than `logging.getLogger(..).warning`. This makes them show up like other warnings, and allows them to be suppressed with [filterwarnings](https://docs.python.org/3/library/warnings.html#warnings.filterwarnings).
- The new NUTS method is only supported on Python 3.3 and newer; a warning will be emitted when importing PINTS in older versions.
- The installation instructions have been updated to reflect that PINTS in now pip-installable.
- The `pints.Logger` can now deal with `None` being logged in place of a proper value.
### Deprecated
- The method `pints.rhat_all_params` was accidentally removed in 0.3.0, but is now back in deprecated form.
### Fixed
- The output of the method `pints.HalfCauchyLogPrior.sample` had the wrong shape [#1192](https://github.com/pints-team/pints/issues/1192).


## [0.3.0] - 2020-08-08
- This is the first pip installable release. The changelog starts documents all changes since this release.
