"""Testing the GOAT optimization model."""

import numpy as np
import pytest

from paraqeet.measurement.state_transfer_fidelity import StateTransferFidelity
from paraqeet.measurement.unitary_fidelity import UnitaryFidelity
from paraqeet.model.drive import DriveOperator
from paraqeet.model.master_equation import MasterEquation
from paraqeet.model.qubit import Qubit
from paraqeet.model.schroedinger_equation import SchroedingerEquation
from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.scipy_optimizer import ScipyOptimizer
from paraqeet.optimizers.scipy_optimizer_gradient import ScipyOptimizerGradient
from paraqeet.propagation.scipy_expm_goat import ScipyExpmGOAT
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import FlatTopGaussianEnvelope
from paraqeet.signal.iq_mixer import IQMixer

FREQ = 4.327884e9 * 2 * np.pi
T_FINAL = 13e-9

RES = 100e9
T1 = Quantity(10e-6, 1e-6, 100e-6)
TEMP = Quantity(10e-3, 1e-3, 50e-3)
T2STAR = Quantity(20e-6, 1e-6, 100e-6)


@pytest.fixture
def tone():
    """Return a cosine tone with a fixed error-function shaped envelope."""
    env = FlatTopGaussianEnvelope()
    env._t_up.set_value(T_FINAL / 5)
    env._t_down.set_value(4 * T_FINAL / 5)
    env._ramp_time.set_value(T_FINAL / 10)
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
    drive = DriveOperator(gen, is_longitudinal=False)
    controlled_qubit = Qubit(Quantity(FREQ, FREQ / 4, FREQ), drives=[drive], t1=T1, temp=TEMP, t2star=T2STAR)
    if request.param == "openSystem":
        model = MasterEquation(controlled_qubit)
    elif request.param == "closedSystem":
        model = SchroedingerEquation(controlled_qubit)
    return ScipyExpmGOAT(model=model, resolution=RES)


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
    )


@pytest.fixture
def gates(prop):
    """Compare the propagator with a gate via the L2 norm."""
    if prop.is_open:
        pytest.skip("Gate optimization is only implemented for closed system.")
    pauli_x = np.array([[0.0, 1], [1, 0.0]])
    prop.set_initial_state(np.identity(2))
    return UnitaryFidelity(
        propagation=prop,
        gate=pauli_x,
    )


@pytest.fixture
def opt_map(gen):
    """Create an optimization map."""
    params = gen.get_parameters()
    params[0].set_value(0.5 * np.pi / T_FINAL)
    params[-2].set_value(1.01 * FREQ)
    optmap = OptimizationMap()
    # Not optimizing t_up, t_down, ramp_time
    optmap.add(gen, [params[0], params[-2], params[-1]])
    return optmap


@pytest.fixture
def grad_opt(states, opt_map):
    """Create a scipy optimizer gradient object over states."""
    return ScipyOptimizerGradient(measure=states, optimization_map=opt_map)


@pytest.fixture
def grad_gates_opt(gates, opt_map):
    """Create a scipy optimizer gradient object over gates."""
    return ScipyOptimizerGradient(measure=gates, optimization_map=opt_map)


@pytest.fixture
def opt(states, opt_map):
    """Create a scipy optimizer object over states."""
    return ScipyOptimizer(measure=states, optimization_map=opt_map)


@pytest.fixture
def gates_opt(gates, opt_map):
    """Create a scipy optimizer gradient object over gates."""
    return ScipyOptimizer(measure=gates, optimization_map=opt_map)


def test_optim_finite_diff(opt) -> None:
    """Check that the optimization goes below threshold."""
    res = opt.optimize(times=T_FINAL)
    assert res.value < 1e-2


def test_optim_goat(grad_opt) -> None:
    """Check that the optimization goes below threshold."""
    res = grad_opt.optimize(times=T_FINAL)
    assert res.value < 1e-2


def test_optim_gates_finite_diff(gates_opt) -> None:
    """Check that the optimization goes below threshold."""
    res = gates_opt.optimize(times=T_FINAL)
    assert res.value < 1e-2


def test_optim_goat_gates(grad_gates_opt) -> None:
    """Check that the optimization goes below threshold."""
    res = grad_gates_opt.optimize(times=T_FINAL)
    assert res.value < 1e-2
