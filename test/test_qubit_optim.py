"""Testing the qubit optimisations."""

import pytest
import numpy as np

from paraqeet.logger import Logger
from paraqeet.optimisation_map import OptimisationMap
from paraqeet.quantity import Quantity
from paraqeet.measurement.state_transfer_fidelity import StateTransferFidelity
from paraqeet.model.drive_operator import DriveOperator
from paraqeet.model.qubit import Qubit
from paraqeet.propagation.scipy_expm_goat import ScipyExpmGOAT
from paraqeet.optimisers.scipy_optimiser import ScipyOptimiser
from paraqeet.optimisers.cmaes_optimiser import CMAEsOptimiser
from paraqeet.optimisers.bayesian_optimiser import BayesianOptimiser

from paraqeet.model.closed_system import ClosedSystem

from paraqeet.signal.iq_mixer import IQMixer
from paraqeet.signal.envelopes import ConstantEnvelope


TONE = ConstantEnvelope()
GEN = IQMixer(envelopes=[TONE])
PARAMS = GEN.get_parameters()

FREQ = 4.8e9 * 2 * np.pi
t_final = 10e-9

PARAMS[0].set_value(0.8 * np.pi / t_final)
PARAMS[2].set_value(1.01 * FREQ)

DRIVE = DriveOperator(GEN, is_longitudinal=False)
CONTROLLED_QUBIT = Qubit(frequency=Quantity(FREQ, 0.8 * FREQ, 1.2 * FREQ), drives=[DRIVE])
MODEL = ClosedSystem(CONTROLLED_QUBIT)

PROP = ScipyExpmGOAT(MODEL, res=100e9)

INIT = np.array([[1.0], [0]])
TARGET = np.array([[0.0], [1]])
ZEROONE = StateTransferFidelity(
    propagation=PROP,
    initial_state=INIT,
    target_state=TARGET,
    times=np.array([0.0, t_final]),
)


@pytest.fixture
def opt():
    """Create ScipyOptimiser optimiser."""
    optmap = OptimisationMap()
    optmap.add(GEN, [PARAMS[0], PARAMS[2]])
    return ScipyOptimiser(ZEROONE, optimisables=optmap)


@pytest.fixture
def cma_opt():
    """Create CMAEs optimiser."""
    optmap = OptimisationMap()
    optmap.add(GEN, [PARAMS[0], PARAMS[2]])
    return CMAEsOptimiser(ZEROONE, optimisables=optmap)


@pytest.fixture
def bay_opt():
    """Create Bayesian optimiser."""
    optmap = OptimisationMap()
    optmap.add(GEN, [PARAMS[0], PARAMS[2]])
    return BayesianOptimiser(ZEROONE, optimisables=optmap)


def test_optim(opt) -> None:
    """Check that the optimization goes below threshold."""
    opt.logger = Logger()
    res = opt.optimise()
    assert res.value < 1e-4


def test_cma(cma_opt: CMAEsOptimiser) -> None:
    """Check that the optimization goes below threshold."""
    cma_opt.logger = Logger()
    res = cma_opt.optimise()
    assert res.value < 1e-4


def test_baysian(bay_opt: BayesianOptimiser) -> None:
    """Check that the optimization goes below threshold."""
    bay_opt.logger = Logger()
    bay_opt.iterations = 200
    res = bay_opt.optimise()
    assert res.value < 1e-3
