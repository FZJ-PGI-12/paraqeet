"""Testing the GRAPE optimization of a TLS system."""

import numpy as np
import pytest

from paraqeet.measurement.state_transfer_fidelity import (
    StateTransferFidelityGRAPE,
)
from paraqeet.measurement.utils import overlap_density_matrix, overlap_state_vector, overlap_vectorized_density_matrix
from paraqeet.model.master_equation import MasterEquation
from paraqeet.model.qubit import Qubit
from paraqeet.model.rotating_frame import RotatingFrameDrive
from paraqeet.model.schroedinger_equation import SchroedingerEquation
from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.scipy_optimizer_gradient import ScipyOptimizerGradient
from paraqeet.propagation.scipy_expm_grape import ScipyExpmGRAPE
from paraqeet.propagation.utils import (
    convert_dm_to_vec,
    grape_operator_sandwich_function_closed,
    grape_operator_sandwich_function_open,
    lindblad_step,
    reverse_lindblad_step,
    reverse_schrodinger_step,
    schrodinger_step,
)
from paraqeet.propagation.vern7_grape import Vern7GRAPE
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import GaussEnvelope
from paraqeet.signal.pwc_generator import PWCGenerator

T_FINAL = 20e-9
FREQ = 1e6
TLIST = np.linspace(0, T_FINAL, 21)
DELTAT = TLIST[1] - TLIST[0]
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


@pytest.fixture(params=["OpenSystem", "ClosedSystem"])
def mode(request):
    return request.param


@pytest.fixture(params=["expm", "ode"])
def solver(request):
    return request.param


@pytest.fixture
def model(pwc_gen, mode):
    init = np.array([[1.0], [0.0]])
    if mode == "OpenSystem":
        init = np.matmul(init, init.T)
        init = convert_dm_to_vec(init)

    drive = RotatingFrameDrive(pwc_gen)
    controlled_qubit = Qubit(Quantity(FREQ, FREQ / 4, FREQ), drives=[drive], t1=T1, temp=TEMP, t2star=T2STAR)
    if mode == "OpenSystem":
        model = MasterEquation(
            hamiltonian_func=controlled_qubit.get_hamiltonian,
            hamiltonian_and_gradient_func=controlled_qubit.get_hamiltonian_and_gradient,
            jump_operators=controlled_qubit.get_jump_operators(),
        )
    elif mode == "ClosedSystem":
        model = SchroedingerEquation(
            hamiltonian_func=controlled_qubit.get_hamiltonian,
            hamiltonian_and_gradient_func=controlled_qubit.get_hamiltonian_and_gradient,
        )
    return model


@pytest.fixture
def states(model, mode, solver):
    """Compare the overlap of the initial and final state."""
    init = np.array([[1.0], [0.0]])
    target = np.array([[0.0], [1]])
    overlap_func = overlap_state_vector
    step_func = schrodinger_step
    reverse_step_func = reverse_schrodinger_step
    operator_sandwich_func = grape_operator_sandwich_function_closed

    if mode == "OpenSystem" and solver == "expm":
        init = np.matmul(init, init.T)
        target = np.matmul(target, target.T)
        init = convert_dm_to_vec(init)
        target = convert_dm_to_vec(target)
        overlap_func = overlap_vectorized_density_matrix
    elif mode == "OpenSystem" and solver == "ode":
        init = np.matmul(init, init.T)
        target = np.matmul(target, target.T)
        overlap_func = overlap_density_matrix
        step_func = lindblad_step
        reverse_step_func = reverse_lindblad_step
        operator_sandwich_func = grape_operator_sandwich_function_open

    if solver == "expm":
        prop_method = ScipyExpmGRAPE(
            eom_func=model.get_value,
            eom_and_grad_func=model.get_value_and_gradient,
            resolution=1 / DELTAT,
            initial_state=init,
            target_state=target,
            operator_sandwich_function=grape_operator_sandwich_function_closed,
        )

    elif mode == "ClosedSystem" and solver == "ode":
        prop_method = Vern7GRAPE(
            eom_func=model.get_value,
            eom_and_gradient_func=model.get_value_and_gradient,
            resolution=10e9,
            initial_state=init,
            target_state=target,
            step_function=step_func,
            reverse_step_function=reverse_step_func,
            operator_sandwich_function=operator_sandwich_func,
        )

    elif mode == "OpenSystem" and solver == "ode":
        prop_method = Vern7GRAPE(
            eom_func=model.get_eom_ode_propagation,
            eom_and_gradient_func=model.get_eom_and_gradient_ode_propagation,
            resolution=10e9,
            initial_state=init,
            target_state=target,
            step_function=step_func,
            reverse_step_function=reverse_step_func,
            operator_sandwich_function=operator_sandwich_func,
            jump_operators=model.jump_operators,
        )

    prop_method.inital_state = init
    prop_method.target_state = target

    return StateTransferFidelityGRAPE(
        propagation_func=prop_method.propagate,
        propagation_and_gradient_func=prop_method.get_value_and_gradient,
        target_state=target,
        overlap=overlap_func,
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
    return ScipyOptimizerGradient(measure_and_gradient_func=states.get_value_and_gradient, optimization_map=opt_map)


def test_optim_grape(opt) -> None:
    """Check that the optimization goes below threshold."""
    res = opt.optimize(times=TLIST)
    assert res.value < 1e-2
