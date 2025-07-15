<div align="center">
  <center><img src="https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/raw/main/big_logo.png" alt="paraqeet_logo" width="90%"/></center>
</div>

# ParaQeet -  A quantum optimal control toolkit with simple parameter management

[![pipeline status](https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/badges/main/pipeline.svg)](https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/commits/main)
[![coverage report](https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/badges/main/coverage.svg)](https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/commits/main)
[![Latest Release](https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/badges/release.svg)](https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/releases)
[![readthedocs](https://readthedocs.org/projects/paraqeet/badge/?version=latest)](https://paraqeet.readthedocs.io/en/latest/)
[![ruff](https://camo.githubusercontent.com/051a04ae958f4a1a5d6444df4cdc520305eef93d5028e6d4c7cd16efa3136cd4/68747470733a2f2f696d672e736869656c64732e696f2f656e64706f696e743f75726c3d68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f61737472616c2d73682f727566662f6d61696e2f6173736574732f62616467652f76322e6a736f6e)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI - Version](https://img.shields.io/pypi/v/paraqeet)](https://pypi.org/project/paraqeet/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/paraqeet)]((https://pypi.org/project/paraqeet/))

*Note: This is a preview version, a 1.0 release is forthcoming.*

Choose a pulse parametrisation, simulate a quantum system, and optimise. 

Combining Quantum Optimal Control methods with automatic differentiation with JAX.
Aimed at resource efficient computation.

We use a top-down approach to make the codebase modular. 
Each module interacts only with the module above it in hierarchy. 

<div align="center">
  <center><img src="https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/raw/main/doc/layers.png" alt="Layers" width="60%"/></center>
</div>

Currently implementated optimization methods
- GRAPE: Gradient Ascent Pulse Enginnering
- GOAT: Gradient Optimization of Analytic conTrols

## Installation from PyPi
Install with `pip install paraqeet`.

## Installation from source
Follow the [contributing](CONTRIBUTING.md) document.
