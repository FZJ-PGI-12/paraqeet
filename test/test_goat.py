import pytest
import numpy as np

from cthree.OptimisationMap import OptimisationMap
from cthree.measurement.StateTransferFidelity import StateTransferFidelity
from cthree.measurement.UnitaryFidelity import UnitaryFidelity

from cthree.propagation.ScipyExpmGOAT import ScipyExpmGOAT

from cthree.optimisers.ScipyOptimiser import ScipyOptimiser
from cthree.optimisers.ScipyOptimiserGradient import ScipyOptimiserGradient

from cthree.model.ClosedModel import ClosedModel
from cthree.model.Hamiltonian import Hamiltonian

from cthree.signal.SimpleGenerator import CosGenerator

from cthree.signal.Device import CosToneErf

from cthree.signal.Device import CosToneErfAD
from cthree.measurement.StateTransferFidelity import StateTransferFidelityAD

FREQ = 4.8e9 * 2 * np.pi
T_FINAL = 10e-9
RES = 100e9


@pytest.fixture
def tone():
    return CosToneErf()


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
    init = np.array([1.0, 0.0j])
    target = np.array([0.0j, 1])
    return StateTransferFidelity(
        propagation=prop,
        initialState=init,
        targetState=target,
        times=np.array([0.0, T_FINAL]),
    )


@pytest.fixture
def gates(prop):
    xGate = np.array([[0.0, 1], [1, 0.0]])
    prop.setInitialState(np.identity(2))
    return UnitaryFidelity(
        propagation=prop,
        gate=xGate,
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
def gradGatesOpt(gates, optMap):
    return ScipyOptimiserGradient(measure=gates, optimisables=optMap)


@pytest.fixture
def opt(states, optMap):
    return ScipyOptimiser(measure=states, optimisables=optMap)


def test_optim_finite_diff(opt) -> None:
    """
    Check that the optimization goes below threshold.
    """
    res = opt.optimise()
    assert res.fun < 1e-4


def test_optim_GOAT(gradOpt) -> None:
    """
    Check that the optimization goes below threshold.
    """
    res = gradOpt.optimise()
    assert res.fun < 1e-4


def test_optim_GOAT_gates(gradGatesOpt) -> None:
    """
    Check that the optimization goes below threshold.
    """
    res = gradGatesOpt.optimise()
    assert res.fun < 1e-4


@pytest.fixture
def toneAD():
    return CosToneErfAD()


@pytest.fixture
def genAD(toneAD):
    genAD = CosGenerator(devices=[toneAD])
    return genAD


@pytest.fixture
def propAD(genAD):
    sigmaZ = np.array([[1.0, 0], [0, -1]])
    sigmaX = np.array([[0.0, 1], [1, 0]])
    drift = FREQ / 2 * sigmaZ
    controlled_qubit = Hamiltonian(subsystems=[drift], drives=[sigmaX], generator=genAD)
    model = ClosedModel(controlled_qubit)
    return ScipyExpmGOAT(model=model, res=RES)


@pytest.fixture
def statesAD(propAD):
    init = np.array([[1.0], [0.0j]])
    target = np.array([[0.0j], [1]])
    return StateTransferFidelityAD(
        propagation=propAD,
        initialState=init,
        targetState=target,
        times=np.array([0.0, T_FINAL]),
    )


@pytest.fixture
def optMapAD(toneAD):
    params = toneAD.getParameters()[0:2]
    params[0].setValue(0.8 * np.pi / T_FINAL)
    params[1].setValue(0.95 * FREQ)
    optmap = OptimisationMap()
    optmap.add(toneAD, params)
    return optmap


@pytest.fixture
def gradOptAD(statesAD, optMapAD):
    return ScipyOptimiserGradient(measure=statesAD, optimisables=optMapAD)


def test_optim_GOAT_AD(gradOptAD) -> None:
    """
    Check that the optimization goes below threshold.
    """
    res = gradOptAD.optimise()
    assert res.fun < 1e-4
