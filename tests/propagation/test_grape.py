"""Test that the GRAPE method works with any propagation method.

Here the gradients are compared with lower tolerances as GRAPE provides an approximation to
the exact gradients.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from paraqeet.eom.master_equation import MasterEquation
from paraqeet.eom.schroedinger_equation import SchroedingerEquation
from paraqeet.exceptions import ConfigurationException
from paraqeet.hamiltonian.drive import Drive
from paraqeet.hamiltonian.qubit import Qubit, QubitHamiltonian
from paraqeet.optimization_map import OptimizationMap
from paraqeet.propagation.auto_diff_gradients import AutoDiffGradients
from paraqeet.propagation.expm import Expm
from paraqeet.propagation.grape import GRAPE
from paraqeet.propagation.utils import (
    convert_dm_to_vec,
    convert_vec_to_dm,
    grape_operator_sandwich_function_closed,
    grape_operator_sandwich_function_open,
    lindblad_step,
    reverse_lindblad_step,
    schrodinger_step,
)
from paraqeet.propagation.vern7 import Vern7
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
# Decay times of the order of the gate time. The backward propagation of GRAPE only shows whether
# it treats the dissipator correctly if the dissipation is not negligible over the gate.
T1_FAST = Quantity(10e-9, 1e-9, 100e-6)
T2STAR_FAST = Quantity(10e-9, 1e-9, 100e-6)

INIT_STATE = np.array([[1.0], [0.0]], dtype=np.complex128)
TARGET_STATE = np.array([[0.0], [1.0]], dtype=np.complex128)


@pytest.fixture
def pwc_generator():
    """Return a piecewise constant generator for a Gaussian pulse."""
    tone = GaussEnvelope(amplitude=Quantity(np.pi / T_FINAL / 3, -np.pi / T_FINAL, np.pi / T_FINAL))
    tone.t_final.set_value(T_FINAL)
    generator = PWCGenerator(envelopes=[tone], tlist=TLIST)
    generator.multiply_flat_top = True
    generator.max_amplitude = 2e8
    return generator


@pytest.fixture
def qubit_hamiltonian(pwc_generator):
    """Return a driven qubit Hamiltonian whose pulse parameters are optimizable."""
    hamiltonian = QubitHamiltonian(Quantity(FREQ, FREQ / 4, FREQ), drives=[])
    hamiltonian.drives = [Drive(hamiltonian.sigma_minus, pwc_generator, add_hermitian=True)]

    optmap = OptimizationMap()
    optmap.add(pwc_generator)
    optmap.register_params_with_optimizables()
    return hamiltonian


@pytest.fixture
def schroedinger(qubit_hamiltonian):
    """Return the Schrödinger equation of the driven qubit."""
    return SchroedingerEquation(
        hamiltonian_func=qubit_hamiltonian.get_value,
        hamiltonian_gradient_func=qubit_hamiltonian.get_gradient,
    )


@pytest.fixture
def master_equation(qubit_hamiltonian):
    """Return the Lindblad master equation of the driven qubit."""
    open_qubit = Qubit(hamiltonian=qubit_hamiltonian, t1=T1, temp=TEMP, t2star=T2STAR)
    return MasterEquation(
        hamiltonian_func=qubit_hamiltonian.get_value,
        hamiltonian_gradient_func=qubit_hamiltonian.get_gradient,
        jump_operators=open_qubit.get_jump_operators(),
    )


@pytest.fixture
def dissipative_master_equation(qubit_hamiltonian):
    """Return the master equation of a driven qubit that decays on the time scale of the gate."""
    open_qubit = Qubit(hamiltonian=qubit_hamiltonian, t1=T1_FAST, temp=TEMP, t2star=T2STAR_FAST)
    return MasterEquation(
        hamiltonian_func=qubit_hamiltonian.get_value,
        hamiltonian_gradient_func=qubit_hamiltonian.get_gradient,
        jump_operators=open_qubit.get_jump_operators(),
    )


def _closed_system(n_pieces):
    """Return the time grid and the Schroedinger equation of the driven qubit on a finer grid.

    The fixtures above are tied to ``TLIST``. Convergence in the width of a pulse piece needs the
    same system on several grids, so this rebuilds it for a given number of pieces.
    """
    tlist = np.linspace(0, T_FINAL, n_pieces + 1)
    tone = GaussEnvelope(amplitude=Quantity(np.pi / T_FINAL / 3, -np.pi / T_FINAL, np.pi / T_FINAL))
    tone.t_final.set_value(T_FINAL)
    generator = PWCGenerator(envelopes=[tone], tlist=tlist)
    generator.multiply_flat_top = True
    generator.max_amplitude = 2e8

    hamiltonian = QubitHamiltonian(Quantity(FREQ, FREQ / 4, FREQ), drives=[])
    hamiltonian.drives = [Drive(hamiltonian.sigma_minus, generator, add_hermitian=True)]
    optmap = OptimizationMap()
    optmap.add(generator)
    optmap.register_params_with_optimizables()

    return tlist, SchroedingerEquation(
        hamiltonian_func=hamiltonian.get_value,
        hamiltonian_gradient_func=hamiltonian.get_gradient,
    )


def _exact_overlap_gradient(propagation, schroedinger, tlist):
    """Return the derivative of the overlap with the target state, without any expansion.

    Automatic differentiation differentiates the propagation itself, so it carries no truncation
    of its own and is the result that the GRAPE expansion converges to. It returns the derivative
    of the propagated state, which the overlap with the target state turns into the same scalar
    that the summed GRAPE gradient approximates.
    """
    autodiff = AutoDiffGradients(propagation, eom_gradient_func=schroedinger.get_gradient)
    state_gradient = np.array(autodiff.get_gradient(tlist))[-1, 0]
    return complex((TARGET_STATE.conj().T @ state_gradient)[0, 0])


def _truncation_error(n_pieces, order):
    """Return the deviation of the GRAPE gradient from the exact derivative of the propagator."""
    tlist, schroedinger = _closed_system(n_pieces)
    dt = tlist[1] - tlist[0]
    propagation = Expm(eom_func=schroedinger.get_value, resolution=1 / dt, initial_state=INIT_STATE)
    grape = GRAPE(
        propagation,
        eom_gradient_func=schroedinger.get_gradient,
        target_state=TARGET_STATE,
        operator_sandwich_function=grape_operator_sandwich_function_closed,
        order=order,
    )
    gradient = np.sum(np.array(grape.get_gradient(tlist))[0])
    exact = _exact_overlap_gradient(propagation, schroedinger, tlist)
    return np.abs(gradient - exact) / np.abs(exact)


def _central_difference(parameter, overlap_func):
    """Return the central difference of ``overlap_func`` w.r.t. ``parameter``."""
    value = np.reshape(np.array(parameter.get_value()), (-1,))
    epsilon = float(1e-6 * np.abs(value[0]))
    parameter.set_value(value + epsilon)
    plus = overlap_func()
    parameter.set_value(value - epsilon)
    minus = overlap_func()
    parameter.set_value(value)
    return (plus - minus) / (2 * epsilon)


def test_grape_expm_against_finite_differences(schroedinger, pwc_generator):
    """Compare GRAPE gradients with FD."""
    propagation = Expm(eom_func=schroedinger.get_value, resolution=1 / DELTAT, initial_state=INIT_STATE)
    grape = GRAPE(
        propagation,
        eom_gradient_func=schroedinger.get_gradient,
        target_state=TARGET_STATE,
        operator_sandwich_function=grape_operator_sandwich_function_closed,
    )

    _, gradient = grape.get_value_and_gradient(TLIST)

    def overlap():
        return complex((TARGET_STATE.conj().T @ np.array(propagation.get_value(TLIST)[-1]))[0, 0])

    parameter = pwc_generator.get_parameters()[0]
    finite_difference = _central_difference(parameter, overlap)

    np.testing.assert_allclose(np.sum(np.array(gradient)[0]), finite_difference, rtol=1e-2)


def test_grape_expm_matches_automatic_differentiation(schroedinger):
    """GRAPE agrees with automatic differentiation of the same propagation.

    The two share the propagation and nothing else: GRAPE propagates the target state backwards
    and expands the derivative of every piece propagator, automatic differentiation differentiates
    the propagation itself.
    """
    propagation = Expm(eom_func=schroedinger.get_value, resolution=1 / DELTAT, initial_state=INIT_STATE)
    grape = GRAPE(
        propagation,
        eom_gradient_func=schroedinger.get_gradient,
        target_state=TARGET_STATE,
        operator_sandwich_function=grape_operator_sandwich_function_closed,
    )

    value, gradient = grape.get_value_and_gradient(TLIST)
    exact = _exact_overlap_gradient(propagation, schroedinger, TLIST)

    assert np.abs(gradient).max() > 1e-12
    np.testing.assert_allclose(value, np.array(propagation.get_value(TLIST)), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.sum(np.array(gradient)[0]), exact, rtol=3e-3)


def test_grape_vern7_matches_grape_expm(schroedinger):
    """The same GRAPE class works with the Vern7 ODE solver and agrees with Expm."""
    expm_grape = GRAPE(
        Expm(eom_func=schroedinger.get_value, resolution=10e9, initial_state=INIT_STATE),
        eom_gradient_func=schroedinger.get_gradient,
        target_state=TARGET_STATE,
        operator_sandwich_function=grape_operator_sandwich_function_closed,
    )
    vern7_grape = GRAPE(
        Vern7(
            eom_func=schroedinger.get_value,
            resolution=10e9,
            initial_state=INIT_STATE,
            step_function=schrodinger_step,
        ),
        eom_gradient_func=schroedinger.get_gradient,
        target_state=TARGET_STATE,
        operator_sandwich_function=grape_operator_sandwich_function_closed,
    )

    expm_value, expm_gradient = expm_grape.get_value_and_gradient(TLIST)
    vern7_value, vern7_gradient = vern7_grape.get_value_and_gradient(TLIST)

    assert np.abs(expm_gradient).max() > 1e-12

    # The tolerance is set by the second order midpoint rule of Expm on a piecewise constant pulse.
    np.testing.assert_allclose(vern7_value, expm_value, rtol=1e-2, atol=1e-4)
    np.testing.assert_allclose(vern7_gradient, expm_gradient, rtol=1e-2, atol=1e-2 * np.abs(expm_gradient).max())


def test_grape_vern7_open_system(master_equation, pwc_generator):
    """Test GRAPE for an open system through the vectorized Lindblad superoperator."""
    init_vec = convert_dm_to_vec(np.matmul(INIT_STATE, INIT_STATE.conj().T))
    target_vec = convert_dm_to_vec(np.matmul(TARGET_STATE, TARGET_STATE.conj().T))
    target_dm = np.matmul(TARGET_STATE, TARGET_STATE.conj().T)

    propagation = Vern7(
        eom_func=master_equation.get_value,
        resolution=10e9,
        initial_state=init_vec,
        step_function=schrodinger_step,
    )
    grape = GRAPE(
        propagation,
        eom_gradient_func=master_equation.get_gradient,
        target_state=target_vec,
        operator_sandwich_function=grape_operator_sandwich_function_closed,
    )

    _, gradient = grape.get_value_and_gradient(TLIST)
    assert np.abs(gradient).max() > 1e-12

    def overlap():
        final_dm = convert_vec_to_dm(np.array(propagation.get_value(TLIST)[-1]))
        return complex(np.trace(target_dm @ np.array(final_dm)))

    parameter = pwc_generator.get_parameters()[0]
    finite_difference = _central_difference(parameter, overlap)

    np.testing.assert_allclose(np.sum(np.array(gradient)[0]), finite_difference, rtol=1e-2)


def _dissipative_grape(master_eq, reverse_step_function, order=2):
    """Return the Vern7 propagation and GRAPE of a Lindblad equation in density matrix form.

    Unlike the superoperator form, the ODE solver keeps the density matrix as a matrix and adds
    the dissipator through its step function and the collapse operators.
    """
    propagation = Vern7(
        eom_func=master_eq.get_eom_ode_propagation,
        resolution=10e9,
        initial_state=np.matmul(INIT_STATE, INIT_STATE.conj().T),
        step_function=lindblad_step,
        jump_operators=master_eq.jump_operators,
    )
    grape = GRAPE(
        propagation,
        eom_gradient_func=master_eq.get_eom_gradient_ode_propagation,
        target_state=np.matmul(TARGET_STATE, TARGET_STATE.conj().T),
        operator_sandwich_function=grape_operator_sandwich_function_open,
        reverse_step_function=reverse_step_function,
        order=order,
    )
    return propagation, grape


def test_grape_vern7_open_system_density_matrix(dissipative_master_equation, pwc_generator):
    """Test GRAPE for a strongly dissipative open system using ODE solvers."""
    propagation, grape = _dissipative_grape(dissipative_master_equation, reverse_lindblad_step)
    target_dm = np.matmul(TARGET_STATE, TARGET_STATE.conj().T)

    _, gradient = grape.get_value_and_gradient(TLIST)

    def overlap():
        return complex(np.trace(target_dm @ np.array(propagation.get_value(TLIST)[-1])))

    parameter = pwc_generator.get_parameters()[0]
    finite_difference = _central_difference(parameter, overlap)

    np.testing.assert_allclose(np.sum(np.array(gradient)[0]), finite_difference, rtol=5e-2)


def test_grape_backward_propagation_uses_the_reverse_step_function(dissipative_master_equation):
    """Test if the reverse step function reaches the backward propagation.

    The forward pass compiles the propagation before the backward pass runs it. If the reverse
    step function were swapped into the forward propagation object, the compilation cache of
    ``Vern7._propagate``, would keep using the forward step function and the reverse one would
    be ignored without any error.

    Here we test this against a step_function that returns zero.
    """

    def zero_step(state, h, cols, *args, **kwargs):
        return jnp.zeros_like(state)

    _, grape = _dissipative_grape(dissipative_master_equation, reverse_lindblad_step)
    _, ignored = _dissipative_grape(dissipative_master_equation, zero_step)

    gradient = np.array(grape.get_gradient(TLIST))
    # A backward propagation that does not move leaves the target state in every sandwich.
    gradient_of_frozen_backward_pass = np.array(ignored.get_gradient(TLIST))

    assert np.abs(gradient).max() > 1e-12
    difference = np.abs(gradient - gradient_of_frozen_backward_pass).max()
    assert difference > 0.1 * np.abs(gradient).max()


def test_grape_requires_a_reverse_step_function_for_collapse_operators(dissipative_master_equation):
    """A dissipator built from collapse operators cannot be adjointed by the equation of motion."""
    with pytest.raises(ConfigurationException, match="collapse operators"):
        _dissipative_grape(dissipative_master_equation, None)


def test_grape_expansion_converges_with_the_pulse_piece_width():
    """The truncated expansion of the propagator gradient converges with its order.

    Halving the width of a pulse piece divides the deviation from the exact derivative by two at
    first order and by four at second order, the definition of the two orders. The bounds are
    loose because the deviation of the reference itself is not exactly a power law.
    """
    first_order = [_truncation_error(20, order=1), _truncation_error(40, order=1)]
    second_order = [_truncation_error(20, order=2), _truncation_error(40, order=2)]

    assert first_order[0] / first_order[1] > 1.7
    assert second_order[0] / second_order[1] > 3.0
    # Second order is an order of magnitude closer on the same grid.
    assert second_order[0] < first_order[0] / 10


def test_grape_higher_order_improves_the_density_matrix_form(dissipative_master_equation, pwc_generator):
    """The expansion also works where the generator is not a matrix product.

    In density matrix form the powers of the Lindbladian cannot be written down as matrices, since
    its dissipator lives in the step function of the solver. Applying them to the states instead
    is what makes the correction work here.
    """
    target_dm = np.matmul(TARGET_STATE, TARGET_STATE.conj().T)
    propagation, _ = _dissipative_grape(dissipative_master_equation, reverse_lindblad_step)

    def overlap():
        return complex(np.trace(target_dm @ np.array(propagation.get_value(TLIST)[-1])))

    finite_difference = _central_difference(pwc_generator.get_parameters()[0], overlap)

    errors = []
    for order in (1, 2):
        _, grape = _dissipative_grape(dissipative_master_equation, reverse_lindblad_step, order=order)
        gradient = np.sum(np.array(grape.get_gradient(TLIST))[0])
        errors.append(np.abs(gradient - finite_difference) / np.abs(finite_difference))

    assert errors[1] < errors[0] / 5


def test_grape_rejects_an_order_below_one(schroedinger):
    """The expansion starts at the first order term, there is nothing below it."""
    with pytest.raises(ConfigurationException, match="at least one"):
        GRAPE(
            Expm(eom_func=schroedinger.get_value, resolution=1 / DELTAT, initial_state=INIT_STATE),
            eom_gradient_func=schroedinger.get_gradient,
            target_state=TARGET_STATE,
            operator_sandwich_function=grape_operator_sandwich_function_closed,
            order=0,
        )


def test_grape_rejects_a_reverse_step_function_without_a_step_function(schroedinger):
    """Propagation methods that take no step function have no use for a reverse one."""
    with pytest.raises(ConfigurationException, match="does not use a step function"):
        GRAPE(
            Expm(eom_func=schroedinger.get_value, resolution=1 / DELTAT, initial_state=INIT_STATE),
            eom_gradient_func=schroedinger.get_gradient,
            target_state=TARGET_STATE,
            operator_sandwich_function=grape_operator_sandwich_function_closed,
            reverse_step_function=schrodinger_step,
        )
