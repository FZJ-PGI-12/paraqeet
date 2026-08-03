"""Test that automatic differentiation of a propagation gives the right gradients."""

import numpy as np
import pytest

from paraqeet.eom.master_equation import MasterEquation
from paraqeet.eom.schroedinger_equation import SchroedingerEquation
from paraqeet.hamiltonian.drive import Drive
from paraqeet.hamiltonian.qubit import Qubit, QubitHamiltonian
from paraqeet.optimization_map import OptimizationMap
from paraqeet.propagation.auto_diff_gradients import AutoDiffGradients
from paraqeet.propagation.expm import Expm
from paraqeet.propagation.finite_difference_gradients import FiniteDifferenceGradients
from paraqeet.propagation.utils import convert_dm_to_vec, schrodinger_step
from paraqeet.propagation.vern7 import Vern7
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import GaussEnvelope
from tests.propagation.test_common_propagation import make_propagation

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
def tone(drive_amplitude):
    """Return the Gaussian pulse that drives the qubit."""
    tone = GaussEnvelope(amplitude=drive_amplitude)
    tone.t_final.set_value(T_FINAL)
    return tone


@pytest.fixture
def optimization_map(tone, drive_amplitude):
    """Return the optimization map holding the drive amplitude."""
    optmap = OptimizationMap()
    optmap.add(tone, [drive_amplitude])
    optmap.register_params_with_optimizables()
    return optmap


@pytest.fixture
def qubit_hamiltonian(tone, optimization_map):
    """Return a driven qubit Hamiltonian with the drive amplitude as its free parameter."""
    hamiltonian = QubitHamiltonian(Quantity(FREQ, FREQ / 4, FREQ), drives=[])
    hamiltonian.drives = [Drive(hamiltonian.sigma_minus, tone, add_hermitian=True)]
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


def test_autodiff_gradient_against_finite_differences(schroedinger, optimization_map):
    """Test gradient of propagated state from automatic differentiation matches finite differences."""
    propagation = Expm(eom_func=schroedinger.get_value, resolution=1 / DELTAT, initial_state=INIT_STATE)
    autodiff = AutoDiffGradients(propagation, eom_gradient_func=schroedinger.get_gradient)

    _, gradient = autodiff.get_value_and_gradient(TLIST)

    # The gradients are taken w.r.t. the parameter value in physical units.
    finite_difference = FiniteDifferenceGradients(propagation, optimization_map).get_gradient(TLIST)

    assert np.abs(gradient).max() > 1e-12
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-5, atol=1e-12)


def test_autodiff_returns_the_states_of_the_propagation(schroedinger):
    """Test if get_value_and_gradient returns correct value and gradient values."""
    propagation = Expm(eom_func=schroedinger.get_value, resolution=1 / DELTAT, initial_state=INIT_STATE)
    autodiff = AutoDiffGradients(propagation, eom_gradient_func=schroedinger.get_gradient)

    value, gradient = autodiff.get_value_and_gradient(TLIST)

    np.testing.assert_allclose(value, np.array(propagation.get_value(TLIST)), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(autodiff.get_value(TLIST), value, rtol=1e-10, atol=1e-12)
    # The state at the first time point is the initial state, which no pulse parameter can move.
    np.testing.assert_array_equal(gradient[0], np.zeros_like(gradient[0]))


@pytest.mark.parametrize(
    ("method", "tolerance"),
    [
        # The expansion of the exponential solves the same discretization as Expm.
        ("chebyshev", 1e-8),
        ("vern7", 1e-5),
        ("diffrax", 1e-4),
        # Euler is first order in the step size, hence the looser bound.
        ("euler", 1e-2),
    ],
)
def test_autodiff_agrees_across_propagation_methods(schroedinger, method, tolerance):
    """Automatic differentiation follows whichever propagation it wraps.

    Every method solves the same equation of motion, so up to their own accuracy they have to
    return the same gradient. This exercises the reverse mode through the ``jax.lax.scan`` of
    ``Vern7``, ``Euler`` and ``ExpmChebyshev``, and through the Diffrax solve.
    """
    resolution = 100e9
    reference = AutoDiffGradients(
        make_propagation("expm", schroedinger.get_value, resolution, INIT_STATE),
        eom_gradient_func=schroedinger.get_gradient,
    )
    autodiff = AutoDiffGradients(
        make_propagation(method, schroedinger.get_value, resolution, INIT_STATE),
        eom_gradient_func=schroedinger.get_gradient,
    )

    expected = np.array(reference.get_gradient(TLIST))
    assert np.abs(expected).max() > 1e-12

    np.testing.assert_allclose(
        autodiff.get_gradient(TLIST), expected, rtol=tolerance, atol=tolerance * np.abs(expected).max()
    )


def test_autodiff_open_system(master_equation, optimization_map):
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
    finite_difference = FiniteDifferenceGradients(propagation, optimization_map).get_gradient(TLIST)

    assert np.abs(gradient).max() > 1e-12
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-5, atol=1e-12)


def test_autodiff_propagator(schroedinger, optimization_map):
    """Test automatic differentiation for propagator (unitary evolution operator)."""
    identity = np.eye(2, dtype=np.complex128)
    propagation = Expm(eom_func=schroedinger.get_value, resolution=1 / DELTAT, initial_state=identity)
    autodiff = AutoDiffGradients(propagation, eom_gradient_func=schroedinger.get_gradient)

    _, gradient = autodiff.get_value_and_gradient(TLIST)
    finite_difference = FiniteDifferenceGradients(propagation, optimization_map).get_gradient(TLIST)

    assert gradient.shape == (len(TLIST), 1) + identity.shape
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-5, atol=1e-12)
