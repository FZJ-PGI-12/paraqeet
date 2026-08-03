"""Test that automatic differentiation of a propagation gives the right gradients."""

import numpy as np
import pytest

from paraqeet.eom.master_equation import MasterEquation
from paraqeet.eom.schroedinger_equation import SchroedingerEquation
from paraqeet.hamiltonian.drive import Drive
from paraqeet.hamiltonian.qubit import Qubit, QubitHamiltonian
from paraqeet.optimization_map import OptimizationMap
from paraqeet.propagation.auto_diff_gradients import AutoDiffGradients
from paraqeet.propagation.euler import Euler
from paraqeet.propagation.expm import Expm
from paraqeet.propagation.utils import convert_dm_to_vec, schrodinger_step
from paraqeet.propagation.vern7 import Vern7
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import GaussEnvelope

T_FINAL = 20e-9
FREQ = 1e6
TLIST = np.linspace(0, T_FINAL, 6)
DELTAT = TLIST[1] - TLIST[0]
T1 = Quantity(10e-6, 1e-9, 100e-6)
TEMP = Quantity(10e-3, 1e-3, 50e-3)
T2STAR = Quantity(10e-6, 1e-9, 100e-6)

INIT_STATE = np.array([[1.0], [0.0]], dtype=np.complex128)


@pytest.fixture
def drive_amplitude():
    """Return the drive amplitude, the only optimizable parameter of the test model."""
    return Quantity(np.pi / T_FINAL / 3, -np.pi / T_FINAL, np.pi / T_FINAL)


@pytest.fixture
def qubit_hamiltonian(drive_amplitude):
    """Return a driven qubit Hamiltonian with the drive amplitude as its free parameter."""
    tone = GaussEnvelope(amplitude=drive_amplitude)
    tone.t_final.set_value(T_FINAL)

    hamiltonian = QubitHamiltonian(Quantity(FREQ, FREQ / 4, FREQ), drives=[])
    hamiltonian.drives = [Drive(hamiltonian.sigma_minus, tone, add_hermitian=True)]

    optmap = OptimizationMap()
    optmap.add(tone, [drive_amplitude])
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


def _central_difference(parameter, value_func):
    """Return the central difference of ``value_func`` w.r.t. ``parameter``, in physical units."""
    value = np.reshape(np.array(parameter.get_value()), (-1,))
    epsilon = float(1e-6 * np.abs(value[0]))
    parameter.set_value(value + epsilon)
    plus = np.array(value_func())
    parameter.set_value(value - epsilon)
    minus = np.array(value_func())
    parameter.set_value(value)
    return (plus - minus) / (2 * epsilon)


def test_autodiff_gradient_against_finite_differences(schroedinger, drive_amplitude):
    """Test gradient of propagated state from automatic differentiation matches finite differences."""
    propagation = Expm(eom_func=schroedinger.get_value, resolution=1 / DELTAT, initial_state=INIT_STATE)
    autodiff = AutoDiffGradients(propagation, eom_gradient_func=schroedinger.get_gradient)

    _, gradient = autodiff.get_value_and_gradient(TLIST)

    # The gradients are taken w.r.t. the parameter value in physical units.
    finite_difference = _central_difference(drive_amplitude, lambda: autodiff.get_value(TLIST))

    assert np.abs(gradient).max() > 1e-12
    np.testing.assert_allclose(gradient[:, 0], finite_difference, rtol=1e-5, atol=1e-12)


def test_autodiff_returns_the_states_of_the_propagation(schroedinger):
    """Test if get_value_and_gradient returns correct value and gradient values."""
    propagation = Expm(eom_func=schroedinger.get_value, resolution=1 / DELTAT, initial_state=INIT_STATE)
    autodiff = AutoDiffGradients(propagation, eom_gradient_func=schroedinger.get_gradient)

    value, gradient = autodiff.get_value_and_gradient(TLIST)

    np.testing.assert_allclose(value, np.array(propagation.get_value(TLIST)), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(autodiff.get_value(TLIST), value, rtol=1e-10, atol=1e-12)
    # The state at the first time point is the initial state, which no pulse parameter can move.
    np.testing.assert_array_equal(gradient[0], np.zeros_like(gradient[0]))


def test_autodiff_agrees_across_propagation_methods(schroedinger):
    """Automatic differentiation follows whichever propagation it wraps.

    Every method solves the same equation of motion, so up to their own accuracy they have to
    return the same gradient. This exercises the reverse mode through the ``jax.lax.scan`` of
    ``Vern7`` and ``Euler`` as well as through the matrix exponential.
    """
    resolution = 100e9
    reference = AutoDiffGradients(
        Expm(eom_func=schroedinger.get_value, resolution=resolution, initial_state=INIT_STATE),
        eom_gradient_func=schroedinger.get_gradient,
    )
    vern7 = AutoDiffGradients(
        Vern7(
            eom_func=schroedinger.get_value,
            resolution=resolution,
            initial_state=INIT_STATE,
            step_function=schrodinger_step,
        ),
        eom_gradient_func=schroedinger.get_gradient,
    )
    euler = AutoDiffGradients(
        Euler(eom_func=schroedinger.get_value, resolution=resolution, initial_state=INIT_STATE),
        eom_gradient_func=schroedinger.get_gradient,
    )

    expected = np.array(reference.get_gradient(TLIST))
    assert np.abs(expected).max() > 1e-12

    np.testing.assert_allclose(vern7.get_gradient(TLIST), expected, rtol=1e-5, atol=1e-5 * np.abs(expected).max())
    # Euler is first order in the step size, hence the looser bound.
    np.testing.assert_allclose(euler.get_gradient(TLIST), expected, rtol=1e-2, atol=1e-2 * np.abs(expected).max())


def test_autodiff_open_system(master_equation, drive_amplitude):
    """Test automatic differentiation works for the vectorized Lindblad superoperator."""
    init_vec = convert_dm_to_vec(np.matmul(INIT_STATE, INIT_STATE.conj().T))
    propagation = Vern7(
        eom_func=master_equation.get_value,
        resolution=10e9,
        initial_state=init_vec,
        step_function=schrodinger_step,
    )
    autodiff = AutoDiffGradients(propagation, eom_gradient_func=master_equation.get_gradient)

    _, gradient = autodiff.get_value_and_gradient(TLIST)
    finite_difference = _central_difference(drive_amplitude, lambda: autodiff.get_value(TLIST))

    assert np.abs(gradient).max() > 1e-12
    np.testing.assert_allclose(gradient[:, 0], finite_difference, rtol=1e-5, atol=1e-12)


def test_autodiff_propagator(schroedinger, drive_amplitude):
    """Test automatic differentiation for propagator (unitary evolution operator)."""
    identity = np.eye(2, dtype=np.complex128)
    propagation = Expm(eom_func=schroedinger.get_value, resolution=1 / DELTAT, initial_state=identity)
    autodiff = AutoDiffGradients(propagation, eom_gradient_func=schroedinger.get_gradient)

    _, gradient = autodiff.get_value_and_gradient(TLIST)
    finite_difference = _central_difference(drive_amplitude, lambda: autodiff.get_value(TLIST))

    assert gradient.shape == (len(TLIST), 1) + identity.shape
    np.testing.assert_allclose(gradient[:, 0], finite_difference, rtol=1e-5, atol=1e-12)
