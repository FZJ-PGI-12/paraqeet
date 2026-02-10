# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Base classes derive from ABC. 
- Force classes to implement abstractmethods.
- Base class Differentiable for classes that provide gradients.
- Mixed derivative evalution in Waveform to fix DRAG gradients.
- Add a CHANGELOG

### Changed

- Adopt US convention for optimizable.
- Refactor inheritance for measurement, optimizer, model, signal, propagation classes.
- Consistent method signatures across the package.
- Double leading underscore to single leading underscore.
- Method signature *_at_one_time() -> *_at_timestep()
- Coupling -> TwoBodyCoupling

### Removed

- `restrict_subsystem` and `_preprocess_vector` from measurement.

## [v0.10.0] - 2025-11-20

### Added

- Adds example notebook (08C) for GOAToverGRAPE method.
- A plotting script for example notebooks.
- Added a dCRAB optimizer.
- FlatTopGaussianFilter instead of multiply_flat_top method.

### Changed

- Extends the GOAToverGRAPE method to support mutiple generators


[unreleased]: https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/compare/main...100-clean-up-inheritance?from_project_id=7369
[v0.10.0]: https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet