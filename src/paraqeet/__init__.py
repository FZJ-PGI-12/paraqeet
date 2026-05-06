"""paraqeet: A quantum optimal control toolkit with simple parameter management."""

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
