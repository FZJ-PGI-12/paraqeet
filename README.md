<div align="center">
  <center><img src="https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/raw/main/big_logo.png" alt="paraqeet_logo" width="90%"/></center>
</div>

# ParaQeet -  A quantum optimal control toolkit with simple parameter management

[![pipeline status](https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/badges/main/pipeline.svg)](https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/commits/main)
[![coverage report](https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/badges/main/coverage.svg)](https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/commits/main)
[![Latest Release](https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/badges/release.svg)](https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/releases)
[![readthedocs](https://readthedocs.org/projects/paraqeet/badge/?version=latest)](https://paraqeet.readthedocs.io/en/latest/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI - Version](https://img.shields.io/pypi/v/paraqeet)](https://pypi.org/project/paraqeet/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/paraqeet.svg)]((https://pypi.org/project/paraqeet/))

*Note: This is a preview version, a 1.0.0 release is forthcoming.*

Choose a pulse parametrisation, simulate a quantum system, and optimize. 

Combining Quantum Optimal Control methods with automatic differentiation with JAX.
Aimed at resource efficient computation.

We use a top-down approach to make the codebase modular. 
Each module interacts only with the module above it in hierarchy. 

<div align="center">
  <center><img src="https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/raw/main/docs/layers.png" alt="Layers" width="60%"/></center>
</div>

Currently implementated optimization methods
- GRAPE: Gradient Ascent Pulse Enginnering
- GOAT: Gradient Optimization of Analytic conTrols
- dCRAB : (Gradient based) dressed Chopped RAndom Basis

## Installation from PyPi
Install with `pip install paraqeet`.

## Installation from source
Follow the [contributing](CONTRIBUTING.md) document.
