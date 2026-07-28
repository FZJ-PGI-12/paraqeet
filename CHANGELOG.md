# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
- Functional interfaces in class constructor, enabling flexibility in model design and optimization.
- Utility functions in each module (instead of static class methods)
- Pedagogical example notebook for gradient evaluation 
- Default callback method to `ScipyOptimizer` for printing optimization progress
- A `Float` type that handles python floats and `jax` Float (Array of float). This can be used for `vmap` purposes.
- Aliases for commonly used classes for easier imports 
- Working example for optimization with QuTiP based objects (with a few wrappers)
- `get_gradient` method to Differentiable classes
- Type Hinting for measurement, Generator classes
- Spellchecking to pre-commit and CI/CD
- Improved documentation with collapsible output cells
- New pages (concepts, code design) to the online documentation
- New dark theme logos
- References in online documentation, including citations in code and notebooks
- Expiry date (of 1 week) to all artifacts during CI/CD
- Missing type annotation and mypy fixes



### Changed
- Constructor of model, propagation, measurement, and optimizer classes
- Combined `StateTransferFidelity` with `StateTransferFidelityAD` classes, such that gradient of fidelity and overlap functions are automatically computed.
- Coupling and Drive classes take operators from the user
- Reduced system sizes in Examples 08B and 08C for faster docs compilation
- System to Hamiltonian. Qubit, Resonator, and Transmon are now containers for the Hamiltonian and other properties.
- Propagation, measurement that inherit from Differentiable expect eom and gradient separately.
- Numpy to google docstring format for maintainability. 
- Updated documentation across the package.
- Rename classes ``ScipyExpm`` -> ``Expm``, ``ScipyExpmGOAT``/``ScipyExpmGRAPE`` -> ``ExpmGOAT``/``ExpmGRAPE``, ``Waveform`` -> ``Signal``
- ``Model`` module split into ``Hamiltonian`` and ``EOM`` modules
- Renamed methods ``propagate`` and ``measure`` -> ``get_value`` for consistency


### Removed
- `Tex-Gyre-Pagella` font from plotting
- `get_value_at_timestep` and `get_value_and_gradient_at_timestep` methods. Replaced with vectorized code
- `DifferentiablePropagation` class in favor of explicitly inheriting from `Differentiable`  
- Solvable in favor of Hamiltonian.

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