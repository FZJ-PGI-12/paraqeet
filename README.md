<div align="center">
  <center><img src="big_logo.png" alt="paraqeet_logo" width="90%"/></center>
</div>

# ParaQeet -  A quantum optimal control toolkit with simple parameter management

[![pipeline status](https://jugit.fz-juelich.de/pgi-12-external/c3-update/badges/main/pipeline.svg)](https://jugit.fz-juelich.de/pgi-12-external/c3-update/-/pipelines)
[![coverage](https://jugit.fz-juelich.de/pgi-12-external/c3-update/badges/main/coverage.svg)](https://jugit.fz-juelich.de/pgi-12-external/c3-update/-/graphs/main/charts)

Choose a pulse parametrisation, simulate a quantum system, and optimise. 

Combining Quantum Optimal Control methods with automatic differentiation with JAX.
Aimed at resource efficient computation.

We use a top-down approach to make the codebase modular. 
Each module interacts only with the module above it in hierarchy. 

<div align="center">
  <center><img src="doc/layers.png" alt="Layers" width="60%"/></center>
</div>

Currently implementated optimization methods
- GRAPE: Gradient Ascent Pulse Enginnering
- GOAT: Gradient Optimization of Analytic conTrols

## Installation
Install with `pip install paraqeet`.
