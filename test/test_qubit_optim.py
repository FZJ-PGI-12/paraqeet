"""Testing the qubit optimisations."""

import pytest
import numpy as np

from cthree.OptimisationMap import OptimisationMap
from cthree.Quantity import Quantity
from cthree.measurement.StateTransferFidelity import StateTransferFidelity
from cthree.model.GeneratorDrive import GeneratorDrive
from cthree.model.Qubit import Qubit
from cthree.propagation.ScipyExpmGOAT import ScipyExpmGOAT
from cthree.optimisers.ScipyOptimiser import ScipyOptimiser
from cthree.optimisers.CMAEsOptimiser import CMAEsOptimiser
from cthree.optimisers.BayesianOptimiser import BayesianOptimiser

from cthree.model.ClosedModel import ClosedModel

from cthree.signal.SimpleGenerator import CosGenerator
from cthree.signal.Device import CosTone


tone = CosTone()
gen = CosGenerator(devices=[tone])
params = tone.getParameters()

FREQ = 4.8e9 * 2 * np.pi
t_final = 10e-9

params[0].setValue(0.8 * np.pi / t_final)
params[1].setValue(1.01 * FREQ)

drive = GeneratorDrive(gen, isLongitudinal=False)
controlled_qubit = Qubit(
    frequency=Quantity(FREQ, 0.8 * FREQ, 1.2 * FREQ), drives=[drive]
)
model = ClosedModel(controlled_qubit)

prop = ScipyExpmGOAT(model, res=100e9)

init = np.array([[1.0], [0]])
target = np.array([[0.0], [1]])
zeroone = StateTransferFidelity(
    propagation=prop,
    initialState=init,
    targetState=target,
    times=np.array([0.0, t_final]),
)


@pytest.fixture
def opt():
    """Create ScipyOptimiser optimiser."""
    optmap = OptimisationMap()
    optmap.add(tone, [params[0], params[1]])
    return ScipyOptimiser(zeroone, optimisables=optmap)


@pytest.fixture
def cma_opt():
    """Create CMAEs optimiser."""
    optmap = OptimisationMap()
    optmap.add(tone, [params[0], params[1]])
    return CMAEsOptimiser(zeroone, optimisables=optmap)


@pytest.fixture
def bay_opt():
    """Create Bayesian optimiser."""
    optmap = OptimisationMap()
    optmap.add(tone, [params[0], params[1]])
    return BayesianOptimiser(zeroone, optimisables=optmap)


def test_optim(opt) -> None:
    """Check that the optimization goes below threshold."""
    res = opt.optimise()
    assert res.value < 1e-4


def test_cma(cma_opt: CMAEsOptimiser) -> None:
    """Check that the optimization goes below threshold."""
    res = cma_opt.optimise()
    assert res.value < 1e-4


def test_baysian(bay_opt: BayesianOptimiser) -> None:
    """Check that the optimization goes below threshold."""
    bay_opt.setIterations(200)
    res = bay_opt.optimise()
    assert res.value < 1e-3
