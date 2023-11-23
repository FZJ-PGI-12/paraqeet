import pytest
import numpy as np

from cthree.measurement.StateTransferFidelity import StateTransferFidelity
from cthree.propagation.GOAT import GOAT
from cthree.ScipyOptimiserGradient import ScipyOptimiser

from cthree.model.ClosedModel import ClosedModel
from cthree.model.Hamiltonian import Hamiltonian

from cthree.signal.SimpleGenerator import CosGenerator
from cthree.signal.Device import CosTone


tone = CosTone()
gen = CosGenerator(devices=[tone])
params = tone.getParameters()

FREQ = 4.8e9
t_final = 10e-9
sigmaZ = np.array([[1.0, 0], [0, -1]])
sigmaX = np.array([[0.0, 1], [1, 0]])

drift = FREQ / 2 * sigmaZ

controlled_qubit = Hamiltonian(subsystems=drift, drives=[sigmaX], generator=gen)
model = ClosedModel(controlled_qubit)

prop = GOAT(model=model, initialTimeStep=0.001e-9)

init = np.array([[1.0], [0.0j]])
target = np.array([[0.0j], [1]])
zeroone = StateTransferFidelity(
    propagation=prop,
    initialState=init,
    targetState=target,
    times=np.array([0.0, t_final]),
)


@pytest.fixture
def opt():
    return ScipyOptimiser(zeroone, optimisables=params)


def test_optim(opt) -> None:
    """
    Check that the optimization goes below threshold.
    """
    res = opt.optimise()
    assert res.fun < 1e-4
