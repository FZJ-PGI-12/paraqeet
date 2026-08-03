"""Test that the GOAT method works with any propagation method."""

import numpy as np
import pytest

from paraqeet.eom.master_equation import MasterEquation
from paraqeet.eom.schroedinger_equation import SchroedingerEquation
from paraqeet.hamiltonian.drive import Drive
from paraqeet.hamiltonian.qubit import Qubit, QubitHamiltonian
from paraqeet.optimization_map import OptimizationMap
from paraqeet.propagation.auto_diff_gradients import AutoDiffGradients
from paraqeet.propagation.expm import Expm
from paraqeet.propagation.goat import GOAT
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


def test_goat_delegates_the_propagation(schroedinger, random_state):
    """Test whether GOAT returns correct value.

    ``GOAT.get_value`` has to leave the states of the wrapped propagation untouched, including
    their shape, which the super state of the gradient part changes.
    """
    propagation = Expm(eom_func=schroedinger.get_value, resolution=1 / DELTAT, initial_state=INIT_STATE)
    goat = GOAT(propagation, eom_gradient_func=schroedinger.get_gradient)

    for state in (INIT_STATE, random_state(2), np.eye(2, dtype=np.complex128)):
        propagation.initial_state = state
        states = goat.get_value(TLIST)
        assert states.shape == (len(TLIST),) + state.shape
        np.testing.assert_allclose(states, propagation.get_value(TLIST))


def test_goat_expm_matches_autodiff(schroedinger):
    """GOAT and automatic differentiation give the same gradient for the same propagation."""
    propagation = Expm(eom_func=schroedinger.get_value, resolution=1 / DELTAT, initial_state=INIT_STATE)
    goat = GOAT(propagation, eom_gradient_func=schroedinger.get_gradient)
    autodiff = AutoDiffGradients(propagation, eom_gradient_func=schroedinger.get_gradient)

    _, goat_gradient = goat.get_value_and_gradient(TLIST)
    _, ad_gradient = autodiff.get_value_and_gradient(TLIST)

    np.testing.assert_allclose(goat_gradient, ad_gradient, rtol=1e-8, atol=1e-10)


def test_goat_vern7_matches_goat_expm(schroedinger):
    """The same GOAT class works with the Vern7 ODE solver and agrees with Expm."""
    expm_goat = GOAT(
        Expm(eom_func=schroedinger.get_value, resolution=10e9, initial_state=INIT_STATE),
        eom_gradient_func=schroedinger.get_gradient,
    )
    vern7_goat = GOAT(
        Vern7(
            eom_func=schroedinger.get_value,
            resolution=10e9,
            initial_state=INIT_STATE,
            step_function=schrodinger_step,
        ),
        eom_gradient_func=schroedinger.get_gradient,
    )
    expm_value, expm_gradient = expm_goat.get_value_and_gradient(TLIST)
    vern7_value, vern7_gradient = vern7_goat.get_value_and_gradient(TLIST)

    # Guard against the comparison below passing on gradients that are all zero.
    assert np.abs(expm_gradient).max() > 1e-10

    # The tolerance is set by the second order midpoint rule of Expm.
    # The gradients are many orders of magnitude smaller than the states, so their absolute
    # tolerance is scaled to their own magnitude.
    np.testing.assert_allclose(vern7_value, expm_value, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(vern7_gradient, expm_gradient, rtol=1e-3, atol=1e-3 * np.abs(expm_gradient).max())


def _central_difference(parameter, value_func):
    """Return the central difference of ``value_func`` w.r.t. ``parameter``, in physical units."""
    value = np.reshape(np.array(parameter.get_value()), (-1,))
    epsilon = float(1e-6 * np.abs(value[0]))
    parameter.set_value(value + epsilon)
    plus = value_func()
    parameter.set_value(value - epsilon)
    minus = value_func()
    parameter.set_value(value)
    return (plus - minus) / (2 * epsilon)


def test_goat_vern7_open_system(master_equation, drive_amplitude):
    """GOAT works for an open system through the vectorized Lindblad superoperator."""
    init_vec = convert_dm_to_vec(np.matmul(INIT_STATE, INIT_STATE.conj().T))

    vern7_goat = GOAT(
        Vern7(
            eom_func=master_equation.get_value,
            resolution=10e9,
            initial_state=init_vec,
            step_function=schrodinger_step,
        ),
        eom_gradient_func=master_equation.get_gradient,
    )
    expm_goat = GOAT(
        Expm(
            eom_func=master_equation.get_value,
            resolution=10e9,
            initial_state=init_vec,
        ),
        eom_gradient_func=master_equation.get_gradient,
    )

    vern7_value, vern7_gradient = vern7_goat.get_value_and_gradient(TLIST)
    expm_value, expm_gradient = expm_goat.get_value_and_gradient(TLIST)

    # Guard against the comparison below passing on gradients that are all zero.
    assert np.abs(expm_gradient).max() > 1e-10

    # The tolerance is set by the second order midpoint rule of Expm.
    # The gradients are many orders of magnitude smaller than the states, so their absolute
    # tolerance is scaled to their own magnitude.
    np.testing.assert_allclose(vern7_value, expm_value, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(vern7_gradient, expm_gradient, rtol=1e-3, atol=1e-3 * np.abs(expm_gradient).max())

    # Also verify with finite difference.
    finite_difference = _central_difference(drive_amplitude, lambda: expm_goat.get_value(TLIST))
    np.testing.assert_allclose(expm_gradient[:, 0], finite_difference, rtol=1e-5, atol=1e-12)


def test_goat_gradient_against_finite_differences(schroedinger, drive_amplitude):
    """Check the GOAT gradient of Expm against a central difference of ``get_value``."""
    goat = GOAT(
        Expm(eom_func=schroedinger.get_value, resolution=1 / DELTAT, initial_state=INIT_STATE),
        eom_gradient_func=schroedinger.get_gradient,
    )

    _, gradient = goat.get_value_and_gradient(TLIST)

    # The gradients are taken w.r.t. the parameter value in physical units.
    finite_difference = _central_difference(drive_amplitude, lambda: goat.get_value(TLIST))

    np.testing.assert_allclose(gradient[:, 0], finite_difference, rtol=1e-5, atol=1e-12)
