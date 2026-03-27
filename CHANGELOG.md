# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.11.1] - 2026-03-27

### Added
- New documentation theme - pydata sphinx theme - similar to Matplotlib and Numpy
- Templates for module documentation
- CI/CD pipeline to test docs compilation
- Added CONTRIBUTING.md to the documentation

### Changed
- Streamlined API documentation build using sphinx.autodoc and sphinx.autosummary
- Fixed spellings using codespell


## [v0.11.0] - 2026-02-11

### Added
- Base classes derive from ABC. 
- Force classes to implement abstractmethods.
- Base class Differentiable for classes that provide gradients.
- Mixed derivative evaluation in Waveform to fix DRAG gradients.
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

- Extends the GOAToverGRAPE method to support multiple generators


## [v0.9.1] - 2025-07-03

- First release.


[v0.11.1]: https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/tree/v0.11.1?ref_type=tags
[v0.11.0]: https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/tree/v0.11.0?ref_type=tags
[v0.10.0]: https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/tree/v0.10.0?ref_type=tags
[v0.9.1]: https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/tree/v0.9.1?ref_type=tags