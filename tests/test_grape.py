"""Testing the GRAPE optimization of a TLS system."""

import numpy as np
import pytest

from paraqeet.measurement.state_transfer_fidelity import (
    StateTransferFidelityGRAPE,
)
from paraqeet.model.closed_system import ClosedSystem
from paraqeet.model.open_system import OpenSystem
from paraqeet.model.qubit import Qubit
from paraqeet.model.rotating_frame_drive import RotatingFrameDrive
from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.scipy_optimizer_gradient import ScipyOptimizerGradient
from paraqeet.propagation.scipy_expm_grape import ScipyExpmGRAPE
from paraqeet.propagation.vern7_grape import Vern7GRAPE
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import GaussEnvelope
from paraqeet.signal.pwc_generator import PWCGenerator

T_FINAL = 20e-9
FREQ = 1e6
TLIST = np.linspace(0, T_FINAL, 21)
T1 = Quantity(10e-6, 1e-9, 100e-6)
TEMP = Quantity(10e-3, 1e-3, 50e-3)
T2STAR = Quantity(10e-6, 1e-9, 100e-6)


@pytest.fixture
def tone():
    tone = GaussEnvelope(amplitude=Quantity(np.pi / T_FINAL / 3, -np.pi / T_FINAL, np.pi / T_FINAL))
    tone.t_final.set_value(T_FINAL)
    return tone


@pytest.fixture
def pwc_gen(tone):
    gen = PWCGenerator(envelopes=[tone], tlist=TLIST)
    gen.multiply_flat_top = True
    gen.max_amplitude = 2 * 1e8
    return gen


@pytest.fixture(scope="function", params=["open_system", "closed_system"])
def model(pwc_gen, request):
    drive = RotatingFrameDrive(pwc_gen)
    controlled_qubit = Qubit(Quantity(FREQ, FREQ / 4, FREQ), drives=[drive], t1=T1, temp=TEMP, t2star=T2STAR)
    if request.param == "open_system":
        model = OpenSystem(controlled_qubit)
    elif request.param == "closed_system":
        model = ClosedSystem(controlled_qubit)
    return model


@pytest.fixture(scope="function", params=["expm", "ode"])
def states(model, request):
    """Compare the overlap of the initial and final state."""
    if request.param == "expm":
        prop_method = ScipyExpmGRAPE(model=model, res=1e9)
        if prop_method.is_open:
            pytest.skip("Currently, ScipyExpmGRAPE is not implemented for open system.")
    elif request.param == "ode":
        model.ode_propagation = True
        prop_method = Vern7GRAPE(model=model, res=1e9)

    init = np.array([[1.0], [0.0]])
    target = np.array([[0.0], [1]])

    if prop_method.is_open:
        init = np.matmul(init, init.T)
        target = np.matmul(target, target.T)

    prop_method.set_initial_state(init)
    prop_method.target_state = target

    return StateTransferFidelityGRAPE(
        propagation=prop_method,
        initial_state=init,
        target_state=target,
    )


@pytest.fixture
def opt_map(pwc_gen):
    """Create an optimization map."""
    optmap = OptimizationMap()
    optmap.add(pwc_gen)
    return optmap


@pytest.fixture
def opt(states, opt_map):
    """Create a scipy optimizer gradient object over states."""
    return ScipyOptimizerGradient(measure=states, optimizables=opt_map)


def test_optim_grape(opt) -> None:
    """Check that the optimization goes below threshold."""
    res = opt.optimize(times=TLIST)
    assert res.value < 1e-2
