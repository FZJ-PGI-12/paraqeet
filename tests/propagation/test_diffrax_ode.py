"""Test the Diffrax ODE propagation against the other solvers of the same equation of motion."""

import diffrax
import numpy as np
import pytest

from paraqeet.eom.master_equation import MasterEquation
from paraqeet.eom.schroedinger_equation import SchroedingerEquation
from paraqeet.exceptions import ConfigurationException
from paraqeet.hamiltonian.drive import Drive
from paraqeet.hamiltonian.qubit import Qubit, QubitHamiltonian
from paraqeet.optimization_map import OptimizationMap
from paraqeet.propagation.auto_diff_gradients import AutoDiffGradients
from paraqeet.propagation.diffrax_ode import DiffraxODE
from paraqeet.propagation.expm import Expm
from paraqeet.propagation.finite_difference_gradients import FiniteDifferenceGradients
from paraqeet.propagation.utils import convert_dm_to_vec, convert_vec_to_dm, lindblad_step, schrodinger_step
from paraqeet.propagation.vern7 import Vern7
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import GaussEnvelope

T_FINAL = 20e-9
FREQ = 1e6
TLIST = np.linspace(0, T_FINAL, 6)
RESOLUTION = 10e9
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


def test_diffrax_matches_vern7(schroedinger):
    """The Diffrax solver propagates the same states as Vern7 and keeps the state normalized."""
    diffrax_ode = DiffraxODE(
        eom_func=schroedinger.get_value,
        resolution=RESOLUTION,
        initial_state=INIT_STATE,
        step_function=schrodinger_step,
        samples_per_step=2,
    )
    vern7 = Vern7(
        eom_func=schroedinger.get_value,
        resolution=RESOLUTION,
        initial_state=INIT_STATE,
        step_function=schrodinger_step,
    )

    states = np.array(diffrax_ode.get_value(TLIST))

    assert states.shape == (len(TLIST),) + INIT_STATE.shape
    np.testing.assert_allclose(np.linalg.norm(states, axis=(1, 2)), np.ones(len(TLIST)), rtol=1e-8)
    np.testing.assert_allclose(states, np.array(vern7.get_value(TLIST)), rtol=1e-5, atol=1e-6)


def test_diffrax_adaptive_step_size(schroedinger):
    """An adaptive high order solver agrees with the fixed step size default configuration."""
    reference = Vern7(
        eom_func=schroedinger.get_value,
        resolution=RESOLUTION,
        initial_state=INIT_STATE,
        step_function=schrodinger_step,
    )
    adaptive = DiffraxODE(
        eom_func=schroedinger.get_value,
        resolution=RESOLUTION,
        initial_state=INIT_STATE,
        step_function=schrodinger_step,
        solver=diffrax.Dopri8(),
        stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-12),
        samples_per_step=2,
    )

    np.testing.assert_allclose(
        np.array(adaptive.get_value(TLIST)), np.array(reference.get_value(TLIST)), rtol=1e-5, atol=1e-6
    )


def test_diffrax_open_system(master_equation):
    """The density matrix form with jump operators agrees with the vectorized superoperator."""
    initial_dm = np.matmul(INIT_STATE, INIT_STATE.conj().T)
    density_matrix = DiffraxODE(
        eom_func=master_equation.get_eom_ode_propagation,
        resolution=RESOLUTION,
        initial_state=initial_dm,
        step_function=lindblad_step,
        jump_operators=master_equation.jump_operators,
        samples_per_step=2,
    )
    superoperator = Vern7(
        eom_func=master_equation.get_value,
        resolution=RESOLUTION,
        initial_state=convert_dm_to_vec(initial_dm),
        step_function=schrodinger_step,
    )

    final_dm = np.array(density_matrix.get_value(TLIST))[-1]
    expected = np.array(convert_vec_to_dm(np.array(superoperator.get_value(TLIST))[-1]))

    np.testing.assert_allclose(np.trace(final_dm), 1.0, rtol=1e-8)
    np.testing.assert_allclose(final_dm, expected, rtol=1e-4, atol=1e-6)


def test_diffrax_autodiff_gradient(schroedinger, optimization_map):
    """Automatic differentiation through the Diffrax solve matches finite differences and Expm."""
    propagation = DiffraxODE(
        eom_func=schroedinger.get_value,
        resolution=RESOLUTION,
        initial_state=INIT_STATE,
        step_function=schrodinger_step,
        samples_per_step=2,
    )
    autodiff = AutoDiffGradients(propagation, eom_gradient_func=schroedinger.get_gradient)
    reference = AutoDiffGradients(
        Expm(eom_func=schroedinger.get_value, resolution=RESOLUTION, initial_state=INIT_STATE),
        eom_gradient_func=schroedinger.get_gradient,
    )

    _, gradient = autodiff.get_value_and_gradient(TLIST)
    expected = np.array(reference.get_gradient(TLIST))
    finite_difference = FiniteDifferenceGradients(propagation, optimization_map).get_gradient(TLIST)

    assert np.abs(gradient).max() > 1e-12
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(gradient, expected, rtol=1e-4, atol=1e-4 * np.abs(expected).max())


def test_diffrax_rejects_invalid_sampling(schroedinger):
    """At least one EOM sample per propagation step is needed for the interpolation."""
    with pytest.raises(ConfigurationException):
        DiffraxODE(
            eom_func=schroedinger.get_value,
            resolution=RESOLUTION,
            initial_state=INIT_STATE,
            step_function=schrodinger_step,
            samples_per_step=0,
        )
