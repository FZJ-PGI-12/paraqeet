"""Testing the GOAT optimisation model."""

import numpy as np
import pytest

from paraqeet.model.open_system import OpenSystem
from paraqeet.optimisation_map import OptimisationMap
from paraqeet.quantity import Quantity
from paraqeet.measurement.state_transfer_fidelity import StateTransferFidelity
from paraqeet.measurement.unitary_fidelity import UnitaryFidelity
from paraqeet.model.closed_system import ClosedSystem
from paraqeet.model.drive_operator import DriveOperator
from paraqeet.model.qubit import Qubit
from paraqeet.optimisers.scipy_optimiser import ScipyOptimiser
from paraqeet.optimisers.scipy_optimiser_gradient import ScipyOptimiserGradient
from paraqeet.propagation.scipy_expm_goat import ScipyExpmGOAT
from paraqeet.signal.envelopes import FlatTopGaussianEnvelope
from paraqeet.signal.iq_mixer import IQMixer

FREQ = 4.327884e9 * 2 * np.pi
T_FINAL = 13e-9
RES = 100e9
T1 = Quantity(10e-6, 1e-9, 100e-6)
TEMP = Quantity(10e-3, 1e-3, 50e-3)
T2STAR = Quantity(10e-6, 1e-9, 100e-6)


@pytest.fixture
def tone():
    """Return a cosine tone with a fixed error-function shaped envelope."""
    env = FlatTopGaussianEnvelope()
    env.t_final.set_value(T_FINAL)
    return env


@pytest.fixture
def gen(tone):
    """Generate a cosine tone."""
    gen = IQMixer(envelopes=[tone])
    return gen


@pytest.fixture(scope="function", params=["openSystem", "closedSystem"])
def prop(gen, request):
    """Solve the equation of motion.

    By piecewise exponentation with the scipy package.

    """
    drive = DriveOperator(gen, isLongitudinal=False)
    controlled_qubit = Qubit(Quantity(FREQ, FREQ / 4, FREQ), drives=[drive], t1=T1, temp=TEMP, t2star=T2STAR)
    if request.param == "openSystem":
        model = OpenSystem(controlled_qubit)
    elif request.param == "closedSystem":
        model = ClosedSystem(controlled_qubit)
    return ScipyExpmGOAT(model=model, res=RES)


@pytest.fixture
def states(prop):
    """Compare the overlap of the initial and final state."""
    init = np.array([[1.0], [0.0]])
    target = np.array([[0.0], [1]])

    if prop.is_open:
        init = np.matmul(init, init.T)
        target = np.matmul(target, target.T)

    return StateTransferFidelity(
        propagation=prop,
        initial_state=init,
        target_state=target,
        times=np.array([0.0, T_FINAL]),
    )


@pytest.fixture
def gates(prop):
    """Compare the propagator with a gate via the L2 norm."""
    if prop.is_open:
        pytest.skip("Gate optimisation is only implemented for closed system.")
    xGate = np.array([[0.0, 1], [1, 0.0]])
    prop.set_initial_state(np.identity(2))
    return UnitaryFidelity(
        propagation=prop,
        gate=xGate,
        times=np.array([0.0, T_FINAL]),
    )


@pytest.fixture
def optMap(gen):
    """Create an optimisation map."""
    params = gen.get_parameters()
    params[0].set_value(0.5 * np.pi / T_FINAL)
    params[2].set_value(1.01 * FREQ)
    optmap = OptimisationMap()
    # Not optimizing t_final
    optmap.add(gen, [params[0], params[2], params[3]])
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


@pytest.fixture
def gatesOpt(gates, optMap):
    """Create a scipy optimiser gradient object over gates."""
    return ScipyOptimiser(measure=gates, optimisables=optMap)


def test_optim_finite_diff(opt) -> None:
    """Check that the optimization goes below threshold."""
    res = opt.optimise()
    assert res.value < 1e-2


def test_optim_GOAT(gradOpt) -> None:
    """Check that the optimization goes below threshold."""
    res = gradOpt.optimise()
    assert res.value < 1e-2


def test_optim_gates_finite_diff(gatesOpt) -> None:
    """Check that the optimization goes below threshold."""
    res = gatesOpt.optimise()
    assert res.value < 1e-2


def test_optim_GOAT_gates(gradGatesOpt) -> None:
    """Check that the optimization goes below threshold."""
    res = gradGatesOpt.optimise()
    assert res.value < 1e-2
