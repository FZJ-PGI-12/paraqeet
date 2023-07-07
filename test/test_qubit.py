import pytest
import numpy as np

from cthree.measurement.StateTransferFidelity import StateTransferFidelity
from cthree.propagation.ScipyExpm import ScipyExpm
from cthree.ScipyOptimiser import ScipyOptimiser

from cthree.model.ClosedModel import ClosedModel
from cthree.model.Hamiltonian import Hamiltonian

from cthree.signal.SimpleGenerator import CosGenerator
from cthree.signal.Device import CosTone

from cthree.QuantumState import QuantumState

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

prop = ScipyExpm(model)

init = QuantumState(vec=np.array([[1.0], [0]]), time=0.0)
target = QuantumState(vec=np.array([[0.0], [1]]), time=t_final)
zeroone = StateTransferFidelity(propagation=prop, initialState=init, targetState=target)


@pytest.fixture
def opt():
    return ScipyOptimiser(zeroone, optimisables=params)


def test_optim(opt) -> None:
    """
    Check that the optimization goes below threshold.
    """
    res = opt.optimise()
    assert res.fun < 1e-4
