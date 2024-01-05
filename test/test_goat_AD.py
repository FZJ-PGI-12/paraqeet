import pytest
import numpy as np

from cthree.OptimisationMap import OptimisationMap
from cthree.measurement.StateTransferFidelity import StateTransferFidelityAD

from cthree.propagation.ScipyExpmGOAT import ScipyExpmGOAT
from cthree.ScipyOptimiser import ScipyOptimiser
from cthree.ScipyOptimiserGradient import ScipyOptimiserGradient

from cthree.model.ClosedModel import ClosedModel
from cthree.model.Hamiltonian import Hamiltonian

from cthree.signal.SimpleGenerator import CosGenerator
from cthree.signal.Device import CosToneErfAD

FREQ = 4.8e9 * 2 * np.pi
T_FINAL = 10e-9
RES = 100e9


@pytest.fixture
def tone():
    return CosToneErfAD()


@pytest.fixture
def gen(tone):
    gen = CosGenerator(devices=[tone])
    return gen


@pytest.fixture
def prop(gen):
    sigmaZ = np.array([[1.0, 0], [0, -1]])
    sigmaX = np.array([[0.0, 1], [1, 0]])
    drift = FREQ / 2 * sigmaZ
    controlled_qubit = Hamiltonian(subsystems=[drift], drives=[sigmaX], generator=gen)
    model = ClosedModel(controlled_qubit)
    return ScipyExpmGOAT(model=model, res=RES)


@pytest.fixture
def states(prop):
    init = np.array([[1.0], [0.0j]])
    target = np.array([[0.0j], [1]])
    return StateTransferFidelityAD(
        propagation=prop,
        initialState=init,
        targetState=target,
        times=np.array([0.0, T_FINAL]),
    )


@pytest.fixture
def optMap(tone):
    params = tone.getParameters()[0:2]
    params[0].setValue(0.8 * np.pi / T_FINAL)
    params[1].setValue(0.95 * FREQ)
    optmap = OptimisationMap()
    optmap.add(tone, params)
    return optmap


@pytest.fixture
def gradOpt(states, optMap):
    return ScipyOptimiserGradient(measure=states, optimisables=optMap)


@pytest.fixture
def opt(states, optMap):
    return ScipyOptimiser(measure=states, optimisables=optMap)


def test_optim_finite_diff_with_JAX(opt) -> None:
    """
    Check that the optimization goes below threshold.
    """
    res = opt.optimise()
    assert res.fun < 1e-4


def test_optim_GOAT_with_AD(gradOpt) -> None:
    """
    Check that the optimization goes below threshold.
    """
    res = gradOpt.optimise()
    assert res.fun < 1e-4
