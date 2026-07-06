"""**ParaQeet: A quantum optimal control toolkit with simple parameter management.**

Choose a pulse parametrization, simulate a quantum system, and optimize.
The package is organized in layers, each interacting only with the layer
above it in the hierarchy:

- ``signal``: pulse parametrizations (envelopes, generators, mixers).
- ``model``: Hamiltonians, drives, and equations of motion.
- ``propagation``: solvers of the equation of motion.
- ``measurement``: fidelities and other goal functions.
- ``optimizers``: optimization algorithms (gradient based and gradient free).

All tunable values are represented by `Quantity` objects. The parameters to
optimize are collected in an `OptimizationMap`, which is handed to an
optimizer together with a goal function.

"""

import jax

from paraqeet.measurement.state_transfer_fidelity import StateTransferFidelity, StateTransferFidelityGRAPE
from paraqeet.measurement.unitary_fidelity import UnitaryFidelity
from paraqeet.model.master_equation import MasterEquation
from paraqeet.model.schroedinger_equation import SchroedingerEquation
from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.scipy_optimizer import ScipyOptimizer
from paraqeet.optimizers.scipy_optimizer_gradient import ScipyOptimizerGradient
from paraqeet.propagation.scipy_expm import ScipyExpm
from paraqeet.propagation.scipy_expm_goat import ScipyExpmGOAT
from paraqeet.propagation.scipy_expm_grape import ScipyExpmGRAPE
from paraqeet.quantity import Array, Quantity

jax.config.update("jax_enable_x64", True)

# aliases
OptMap = OptimizationMap
SchrEq = SchroedingerEquation


__all__ = [
    "OptimizationMap",
    "OptMap",
    "Array",
    "Quantity",
    "SchroedingerEquation",
    "SchrEq",
    "MasterEquation",
    "ScipyExpm",
    "ScipyExpmGOAT",
    "ScipyExpmGRAPE",
    "StateTransferFidelity",
    "StateTransferFidelityGRAPE",
    "UnitaryFidelity",
    "ScipyOptimizer",
    "ScipyOptimizerGradient",
]
