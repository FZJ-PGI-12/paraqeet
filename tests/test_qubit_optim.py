"""Testing the qubit optimizations."""

import numpy as np
import pytest

from paraqeet.logger import Logger
from paraqeet.measurement.state_transfer_fidelity import StateTransferFidelity
from paraqeet.model.closed_system import ClosedSystem
from paraqeet.model.drive_operator import DriveOperator
from paraqeet.model.qubit import Qubit
from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.bayesian_optimizer import BayesianOptimizer
from paraqeet.optimizers.cmaes_optimizer import CMAEsOptimizer
from paraqeet.optimizers.scipy_optimizer import ScipyOptimizer
from paraqeet.propagation.scipy_expm_goat import ScipyExpmGOAT
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import ConstantEnvelope
from paraqeet.signal.iq_mixer import IQMixer

TONE = ConstantEnvelope()
GEN = IQMixer(envelopes=[TONE])
PARAMS = GEN.get_parameters()

FREQ = 4.8e9 * 2 * np.pi
T_FINAL = 10e-9

PARAMS[0].set_value(0.8 * np.pi / T_FINAL)
PARAMS[2].set_value(1.01 * FREQ)

DRIVE = DriveOperator(GEN, is_longitudinal=False)
CONTROLLED_QUBIT = Qubit(frequency=Quantity(FREQ, 0.8 * FREQ, 1.2 * FREQ), drives=[DRIVE])
MODEL = ClosedSystem(CONTROLLED_QUBIT)

PROP = ScipyExpmGOAT(MODEL, resolution=100e9)

INIT = np.array([[1.0], [0]])
TARGET = np.array([[0.0], [1]])
ZEROONE = StateTransferFidelity(
    propagation=PROP,
    initial_state=INIT,
    target_state=TARGET,
)


@pytest.fixture
def opt():
    """Create ScipyOptimizer optimizer."""
    optmap = OptimizationMap()
    optmap.add(GEN, [PARAMS[0], PARAMS[2]])
    return ScipyOptimizer(ZEROONE, optimization_map=optmap)


@pytest.fixture
def cma_opt():
    """Create CMAEs optimizer."""
    optmap = OptimizationMap()
    optmap.add(GEN, [PARAMS[0], PARAMS[2]])
    return CMAEsOptimizer(ZEROONE, optimization_map=optmap)


@pytest.fixture
def bay_opt():
    """Create Bayesian optimizer."""
    optmap = OptimizationMap()
    optmap.add(GEN, [PARAMS[0], PARAMS[2]])
    return BayesianOptimizer(ZEROONE, optimization_map=optmap)


def test_optim(opt) -> None:
    """Check that the optimization goes below threshold."""
    opt.logger = Logger()
    res = opt.optimize(times=T_FINAL)
    assert res.value < 1e-4


def test_cma(cma_opt: CMAEsOptimizer) -> None:
    """Check that the optimization goes below threshold."""
    cma_opt.logger = Logger()
    res = cma_opt.optimize(times=T_FINAL)
    assert res.value < 1e-4


def test_baysian(bay_opt: BayesianOptimizer) -> None:
    """Check that the optimization goes below threshold."""
    bay_opt.logger = Logger()
    bay_opt.iterations = 200
    res = bay_opt.optimize(times=T_FINAL)
    assert res.value < 1e-3
