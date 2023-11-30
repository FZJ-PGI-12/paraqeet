import pytest
import numpy as np

from cthree.measurement.StateTransferFidelity import StateTransferFidelity

from cthree.propagation.GOAT import GOAT
from cthree.ScipyOptimiser import ScipyOptimiser
from cthree.ScipyOptimiserGradient import ScipyOptimiserGradient

from cthree.model.ClosedModel import ClosedModel
from cthree.model.Hamiltonian import Hamiltonian

from cthree.signal.SimpleGenerator import CosGenerator
from cthree.signal.Device import CosTone

tone = CosTone()
gen = CosGenerator(devices=[tone])
params = tone.getParameters()

FREQ = 4.8e9 * 2 * np.pi
t_final = 10e-9
params[0].setValue(0.9 * np.pi / t_final)
params[1].setValue(1.001 * FREQ)

sigmaZ = np.array([[1.0, 0], [0, -1]])
sigmaX = np.array([[0.0, 1], [1, 0]])

drift = FREQ / 2 * sigmaZ

controlled_qubit = Hamiltonian(subsystems=[drift], drives=[sigmaX], generator=gen)
model = ClosedModel(controlled_qubit)

# prop = ScipyExpm(model=model, res=100e9)
prop = GOAT(model=model, initialTimeStep=0.01e-9)

init = np.array([[1.0], [0.0j]])
target = np.array([[0.0j], [1]])
zeroone = StateTransferFidelity(
    propagation=prop,
    initialState=init,
    targetState=target,
    times=np.array([0.0, t_final]),
)


@pytest.fixture
def fid():
    return StateTransferFidelity(
        propagation=prop,
        initialState=init,
        targetState=target,
        times=np.array([0.0, t_final]),
    )


@pytest.fixture
def gradOpt(fid):
    return ScipyOptimiserGradient(fid, optimisables=params)


@pytest.fixture
def opt(fid):
    return ScipyOptimiser(fid, optimisables=params)


def test_optim(opt) -> None:
    """
    Check that the optimization goes below threshold.
    """
    res = opt.optimise()
    assert res.fun < 1e-4


def test_optim_grad(gradOpt) -> None:
    """
    Check that the optimization goes below threshold.
    """
    res = gradOpt.optimise()
    assert res.fun < 1e-4
