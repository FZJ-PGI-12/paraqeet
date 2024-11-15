"""Testing the GOAT optimisation model."""

import numpy as np
import pytest

from cthree.OptimisationMap import OptimisationMap
from cthree.Quantity import Quantity
from cthree.measurement.StateTransferFidelity import StateTransferFidelity
from cthree.measurement.StateTransferFidelity import StateTransferFidelityAD
from cthree.measurement.UnitaryFidelity import UnitaryFidelity
from cthree.model.ClosedModel import ClosedModel
from cthree.model.GeneratorDrive import GeneratorDrive
from cthree.model.Qubit import Qubit
from cthree.optimisers.ScipyOptimiser import ScipyOptimiser
from cthree.optimisers.ScipyOptimiserGradient import ScipyOptimiserGradient
from cthree.propagation.ScipyExpmGOAT import ScipyExpmGOAT
from cthree.signal.Envelopes import FlatTopGaussianEnvelope
from cthree.signal.SimpleGenerator import CosGenerator
from test.DummyDevice import CosToneErfAD

FREQ = 4.8e9 * 2 * np.pi
T_FINAL = 10e-9
RES = 100e9


@pytest.fixture
def tone():
    """Return a cosine tone with a fixed error-function shaped envelope."""
    return FlatTopGaussianEnvelope()


@pytest.fixture
def gen(tone):
    """Generate a cosine tone."""
    gen = CosGenerator(envelopes=[tone])
    return gen


@pytest.fixture
def prop(gen):
    """Solve the equation of motion.

    By piecewise exponentation with the scipy package.

    """
    drive = GeneratorDrive(gen, isLongitudinal=False)
    controlled_qubit = Qubit(
        Quantity(FREQ, 0.5 * FREQ, 1.5 * FREQ), drives=[drive]
    )
    model = ClosedModel(controlled_qubit)
    return ScipyExpmGOAT(model=model, res=RES)


@pytest.fixture
def states(prop):
    """Compare the overlap of the initial and final state."""
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
    """Compare the propagator with a gate via the L2 norm."""
    xGate = np.array([[0.0, 1], [1, 0.0]])
    prop.setInitialState(np.identity(2))
    return UnitaryFidelity(
        propagation=prop,
        gate=xGate,
        times=np.array([0.0, T_FINAL]),
    )


@pytest.fixture
def optMap(gen):
    """Create an optimisation map."""
    params = gen.getParameters()
    params[0].setValue(0.8 * np.pi / T_FINAL)
    params[2].setValue(0.96 * FREQ)
    optmap = OptimisationMap()
    optmap.add(gen, params)
    return optmap


@pytest.fixture
def gradOpt(states, optMap):
    """Create a scipy optimiser gradient object over states."""
    return ScipyOptimiserGradient(measure=states, optimisables=optMap)


@pytest.fixture
def gradGatesOpt(gates, optMap):
    """Create a scipy optimiser gradient object over gates."""
    return ScipyOptimiserGradient(measure=gates, optimisables=optMap)


@pytest.fixture
def opt(states, optMap):
    """Create a scipy optimiser object over states."""
    return ScipyOptimiser(measure=states, optimisables=optMap)


def test_optim_finite_diff(opt) -> None:
    """Check that the optimization goes below threshold."""
    res = opt.optimise()
    assert res.value < 1e-4


def test_optim_GOAT(gradOpt) -> None:
    """Check that the optimization goes below threshold."""
    res = gradOpt.optimise()
    assert res.value < 1e-4


def test_optim_GOAT_gates(gradGatesOpt) -> None:
    """Check that the optimization goes below threshold."""
    res = gradGatesOpt.optimise()
    assert res.value < 1e-4


@pytest.fixture
def toneAD():
    """Create a dummy CosToneErf class.

    Without analytical gradients to test AD gradients.

    """
    return CosToneErfAD()


@pytest.fixture
def genAD(toneAD):
    """Generate a cosine tone."""
    genAD = CosGenerator(envelopes=[toneAD])
    return genAD


@pytest.fixture
def propAD(genAD):
    """Create a Scipy piecewise exponentiation GOAT model."""
    drive = GeneratorDrive(genAD, isLongitudinal=False)
    controlled_qubit = Qubit(
        frequency=Quantity(FREQ, 0.8 * FREQ, 1.2 * FREQ), drives=[drive]
    )
    model = ClosedModel(controlled_qubit)
    return ScipyExpmGOAT(model=model, res=RES)


@pytest.fixture
def statesAD(propAD):
    """Return the state transfer fidelity."""
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
    """Return the optimisation map."""
    params = toneAD.getParameters()[0:2]
    params[0].setValue(0.8 * np.pi / T_FINAL)
    params[2].setValue(0.95 * FREQ)
    optmap = OptimisationMap()
    optmap.add(toneAD, params)
    return optmap


@pytest.fixture
def gradOptAD(statesAD, optMapAD):
    """Return the Scipy optimiser gradient object."""
    return ScipyOptimiserGradient(measure=statesAD, optimisables=optMapAD)


def test_optim_GOAT_AD(gradOptAD) -> None:
    """Check that the optimization goes below threshold."""
    res = gradOptAD.optimise()
    assert res.value < 1e-4
