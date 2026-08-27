"""Test the Chebyshev expansion of the matrix exponential."""

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
from paraqeet.propagation.expm_chebyshev import ExpmChebyshev
from paraqeet.propagation.utils import convert_dm_to_vec
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import GaussEnvelope
from tests.model.empty_hamiltonian import EmptyHamiltonian
from tests.propagation.test_common_propagation import check_propagation

T_FINAL = 20e-9
FREQ = 1e6
TLIST = np.linspace(0, T_FINAL, 6)
RESOLUTION = 10e9

INIT_STATE = np.array([[1.0], [0.0]], dtype=np.complex128)


@pytest.fixture
def chebyshev():
    """Return a Chebyshev propagation generating method for the empty Hamiltonian."""

    def _method(dimension, order=16):
        system = EmptyHamiltonian(dimension)
        eom = SchroedingerEquation(
            hamiltonian_func=system.get_value,
            hamiltonian_gradient_func=system.get_gradient,
        )
        return ExpmChebyshev(
            eom_func=eom.get_value,
            resolution=3,
            initial_state=np.eye(dimension, dtype=np.complex128),
            order=order,
        )

    return _method


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
    open_qubit = Qubit(
        hamiltonian=qubit_hamiltonian,
        t1=Quantity(10e-6, 1e-9, 100e-6),
        temp=Quantity(10e-3, 1e-3, 50e-3),
        t2star=Quantity(10e-6, 1e-9, 100e-6),
    )
    return MasterEquation(
        hamiltonian_func=qubit_hamiltonian.get_value,
        hamiltonian_gradient_func=qubit_hamiltonian.get_gradient,
        jump_operators=open_qubit.get_jump_operators(),
    )


def test_order_has_to_be_positive(chebyshev):
    with pytest.raises(ConfigurationException):
        chebyshev(2, order=0)

    propagation = chebyshev(2)
    with pytest.raises(ConfigurationException):
        propagation.order = 0


def test_state_dimension_vector(random_state, chebyshev, ts):
    for _ in range(5):
        dim = np.random.randint(2, 30)
        state = random_state(dim)
        propagation = chebyshev(dim)
        propagation.initial_state = state
        propagated_states = propagation.get_value(ts)
        assert propagated_states.shape[0] == len(ts)
        assert propagated_states.shape[1:] == state.shape


def test_state_dimension_matrix(chebyshev, ts):
    for _ in range(5):
        dim = np.random.randint(2, 10)
        state = np.eye(dim, dtype=np.complex128)
        propagation = chebyshev(dim)
        propagation.initial_state = state
        propagated_states = propagation.get_value(ts)
        assert propagated_states.shape[0] == len(ts)
        assert propagated_states.shape[1:] == state.shape


def test_needs_initial_state(chebyshev):
    for dim in range(2, 5):
        check_propagation(chebyshev(dim), dim)


def test_vanishing_generator_leaves_the_state_untouched(random_state, chebyshev, ts):
    """The empty Hamiltonian has a vanishing radius, which the rescaling has to catch."""
    state = random_state(7)
    propagation = chebyshev(7)
    propagation.initial_state = state
    propagated_states = np.array(propagation.get_value(ts))
    assert np.all(np.isfinite(propagated_states))
    np.testing.assert_allclose(propagated_states[-1], state, rtol=1e-12, atol=1e-14)


def test_propagation_agrees_with_the_matrix_exponential(schroedinger):
    """The expansion of the exponential has to reproduce the dense matrix exponential."""
    chebyshev = ExpmChebyshev(
        eom_func=schroedinger.get_value, resolution=RESOLUTION, initial_state=INIT_STATE, order=16
    )
    expm = Expm(eom_func=schroedinger.get_value, resolution=RESOLUTION, initial_state=INIT_STATE)

    expected = np.array(expm.get_value(TLIST))
    np.testing.assert_allclose(chebyshev.get_value(TLIST), expected, rtol=1e-12, atol=1e-14)


def test_propagation_of_a_propagator(schroedinger):
    """Propagating the identity gives the propagator, which stays unitary."""
    identity = np.eye(2, dtype=np.complex128)
    chebyshev = ExpmChebyshev(eom_func=schroedinger.get_value, resolution=RESOLUTION, initial_state=identity)

    propagators = np.array(chebyshev.get_value(TLIST))
    np.testing.assert_allclose(propagators[-1] @ propagators[-1].conj().T, identity, rtol=1e-10, atol=1e-12)


def test_open_system_agrees_with_the_matrix_exponential(master_equation):
    """The expansion also converges for the generator of the master equation."""
    init_vec = convert_dm_to_vec(np.matmul(INIT_STATE, INIT_STATE.conj().T))
    chebyshev = ExpmChebyshev(eom_func=master_equation.get_value, resolution=RESOLUTION, initial_state=init_vec)
    expm = Expm(eom_func=master_equation.get_value, resolution=RESOLUTION, initial_state=init_vec)

    expected = np.array(expm.get_value(TLIST))
    np.testing.assert_allclose(chebyshev.get_value(TLIST), expected, rtol=1e-10, atol=1e-12)


def test_truncating_the_expansion_is_inaccurate(schroedinger):
    """An order below the radius of the generator does not converge, which motivates the default."""
    low_order = ExpmChebyshev(
        eom_func=schroedinger.get_value, resolution=1 / (TLIST[1] - TLIST[0]), initial_state=INIT_STATE, order=1
    )
    expm = Expm(eom_func=schroedinger.get_value, resolution=1 / (TLIST[1] - TLIST[0]), initial_state=INIT_STATE)

    difference = np.abs(np.array(low_order.get_value(TLIST)) - np.array(expm.get_value(TLIST)))
    assert difference.max() > 1e-6


def test_suggested_order(schroedinger):
    """The suggested order is below the default here, and propagates just as accurately."""
    propagation = ExpmChebyshev(eom_func=schroedinger.get_value, resolution=RESOLUTION, initial_state=INIT_STATE)
    order = propagation.suggested_order(TLIST, tolerance=1e-14, margin=0)
    assert 1 <= order < propagation.order

    expm = Expm(eom_func=schroedinger.get_value, resolution=RESOLUTION, initial_state=INIT_STATE)
    expected = np.array(expm.get_value(TLIST))
    converged = ExpmChebyshev(
        eom_func=schroedinger.get_value, resolution=RESOLUTION, initial_state=INIT_STATE, order=order
    )
    np.testing.assert_allclose(converged.get_value(TLIST), expected, rtol=1e-10, atol=1e-13)


def test_changing_the_order_reaches_the_compiled_gradient(schroedinger):
    """The order travels as the length of an array, so changing it recompiles the propagation."""
    resolution = 1 / (TLIST[1] - TLIST[0])
    propagation = ExpmChebyshev(
        eom_func=schroedinger.get_value, resolution=resolution, initial_state=INIT_STATE, order=1
    )
    autodiff = AutoDiffGradients(propagation, eom_gradient_func=schroedinger.get_gradient)
    expected = np.array(
        AutoDiffGradients(
            Expm(eom_func=schroedinger.get_value, resolution=resolution, initial_state=INIT_STATE),
            eom_gradient_func=schroedinger.get_gradient,
        ).get_gradient(TLIST)
    )

    truncated = np.array(autodiff.get_gradient(TLIST))
    propagation.order = 16
    converged = np.array(autodiff.get_gradient(TLIST))

    assert np.abs(truncated - expected).max() > 1e-6 * np.abs(expected).max()
    np.testing.assert_allclose(converged, expected, rtol=1e-8, atol=1e-10 * np.abs(expected).max())
