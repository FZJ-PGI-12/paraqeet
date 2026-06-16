"""Test the Transmon object."""

import numpy as np
import pytest

from paraqeet.differentiable import Differentiable
from paraqeet.exceptions import ConfigurationException
from paraqeet.model.drive import Drive
from paraqeet.model.master_equation import MasterEquation
from paraqeet.model.transmon import Transmon
from paraqeet.propagation.scipy_expm import ScipyExpm
from paraqeet.propagation.utils import convert_dm_to_vec, lindblad_step
from paraqeet.propagation.vern7 import Vern7
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import FlatTopGaussianEnvelope, ZeroEnvelope
from paraqeet.signal.iq_mixer import IQMixer

DIMS = 3
FREQ = 4.8e9 * 2 * np.pi
ANHARMONICITY = -200e6 * 2 * np.pi
LEN_SIG = 1001
T1 = Quantity(1e-9, 1e-9, 100e-6)
TEMP = Quantity(10e-3, 1e-3, 50e-3)
T2STAR = Quantity(10e-9, 1e-9, 100e-6)


@pytest.fixture
def time_samples():
    """Generate time samples from the given signal length.

    The given signal length `LEN_SIG` is a module level global variable.

    """
    return np.linspace(0, 10e-9, LEN_SIG)


@pytest.fixture
def tone():
    """Return a object for sinusoidal tone generator with error envelopes."""
    return FlatTopGaussianEnvelope()


@pytest.fixture
def gen(tone):
    """Return a sinusoidal tone generator object."""
    gen = IQMixer(envelopes=[tone])
    return gen


@pytest.fixture
def hamiltonian(gen):
    """Return a transmon object."""

    def _method(num_levels):
        transmon = Transmon(
            num_levels=num_levels,
            frequency=Quantity(FREQ, 0.8 * FREQ, 1.2 * FREQ),
            anharmonicity=Quantity(ANHARMONICITY, 1.2 * ANHARMONICITY, 0.8 * ANHARMONICITY),
            drives=[],
        )
        drive_op = transmon.annihilation_op + (transmon.annihilation_op).conj().T
        drive = Drive(drive_op, gen)
        drive.set_optimizable_parameters(drive.get_parameters())
        transmon.drives = [drive]
        return transmon

    return _method


@pytest.fixture
def open_transmon():
    """Return an open model for the transmon."""
    tone = ZeroEnvelope()
    generator = IQMixer(envelopes=[tone])
    transmon = Transmon(
        frequency=Quantity(FREQ, 0.8 * FREQ, 1.2 * FREQ),
        anharmonicity=Quantity(ANHARMONICITY, 1.2 * ANHARMONICITY, 0.8 * ANHARMONICITY),
        drives=[],
        num_levels=DIMS,
    )
    drive_op = transmon.annihilation_op + (transmon.annihilation_op).conj().T
    drive = Drive(drive_op, generator)
    transmon.drives = [drive]

    transmon.t1 = T1
    transmon.temp = TEMP
    transmon.t2star = T2STAR
    model = MasterEquation(
        hamiltonian_func=transmon.get_value,
        hamiltonian_and_gradient_func=transmon.get_value_and_gradient,
        jump_operators=transmon.get_jump_operators(),
    )

    return model


@pytest.fixture
def expm(open_transmon):
    init = np.zeros((DIMS, 1), dtype=np.complex128)
    init[DIMS - 1][0] = 1  # Fully excited state
    init_dm = np.matmul(init, init.T)
    init_dm_vec = convert_dm_to_vec(init_dm)

    prop = ScipyExpm(open_transmon.get_value, resolution=100e9, initial_state=init_dm_vec)
    return prop


@pytest.fixture
def ode(open_transmon):
    init = np.zeros((DIMS, 1), dtype=np.complex128)
    init[DIMS - 1][0] = 1  # Fully excited state
    init_dm = np.matmul(init, init.T)

    prop = Vern7(
        open_transmon.get_eom_ode_propagation,
        resolution=100e9,
        initial_state=init_dm,
        step_function=lindblad_step,
        jump_operators=open_transmon.jump_operators,
    )
    return prop


def test_get_hamiltonian(hamiltonian, time_samples):
    """Test the get_hamiltonian method."""
    for dim in np.arange(1, 10):
        hamil = hamiltonian(dim)
        hams = hamil.get_value(time_samples)
        assert hams.shape == time_samples.shape + (dim, dim)


def test_gradient(gen, hamiltonian, time_samples):
    """Test the gradients of the system.

    The Hamiltonian should have all derivatives of the drive
    plus the derivative w.r.t. the frequency and anharmonicity.

    """
    for dim in np.arange(1, 10):
        hamil = hamiltonian(dim)
        if not isinstance(hamil, Differentiable):
            continue
        hamil.set_optimizable_parameters(hamil.get_parameters())
        _, grads = gen.get_value_and_gradient(time_samples)
        _, ham_grads = hamil.get_value_and_gradient(time_samples)
        assert ham_grads.shape == (grads.shape[0], grads.shape[1] + 2, dim, dim)


def test_get_drive_matrix(hamiltonian, time_samples):
    """Test the get_drive_matrix method of the Hamiltonian."""
    dim = np.random.randint(2, 10)
    hamil = hamiltonian(dim)
    drive_matrix = hamil.get_drive_matrix(time_samples)
    assert drive_matrix.shape == time_samples.shape + (dim, dim)


def test_get_drive_gradients(gen, hamiltonian, time_samples):
    """Test the drive gradients of the Hamiltonian."""
    dim = np.random.randint(2, 10)
    hamil = hamiltonian(dim)
    _, grads = gen.get_value_and_gradient(time_samples)
    drive_gradients = hamil.get_drive_gradients(time_samples)
    assert drive_gradients.shape == (grads.shape[0], grads.shape[1], dim, dim)


def test_decay_expm(expm):
    t_final = 20e-9
    ts = np.linspace(0, t_final, 101)

    states = expm.propagate(ts)
    final_state = states[-1]

    # Check if final state is density matrix
    assert np.isclose(np.linalg.trace(final_state), 1)

    # Check the excited state population
    assert np.isclose(final_state[DIMS - 1, DIMS - 1], 0)

    # Check the ground state population
    assert np.isclose(final_state[0, 0], 1)


def test_decay_ode(ode):
    t_final = 20e-9
    ts = np.linspace(0, t_final, 101)

    states = ode.propagate(ts)
    final_state = states[-1]

    # Check if final state is density matrix
    assert np.isclose(np.linalg.trace(final_state), 1)

    # Check the excited state population
    assert np.isclose(final_state[DIMS - 1, DIMS - 1], 0)

    # Check the ground state population
    assert np.isclose(final_state[0, 0], 1)


def test_setters_and_getters(hamiltonian, random_quantity):
    hamil = hamiltonian(np.random.randint(2, 10))

    frequency = random_quantity(1)
    hamil.frequency = frequency
    assert hamil.frequency == frequency

    anharmonicity = random_quantity(1)
    hamil.anharmonicity = anharmonicity
    assert hamil.anharmonicity == anharmonicity


def test_needs_parameters_for_decay_rates(hamiltonian):
    for _ in range(10):
        for dim in np.arange(1, 10):
            hamil = hamiltonian(dim)
            t1 = Quantity(1.0, 0.0, 2.0) if np.random.random() < 0.8 else None
            t2star = Quantity(1.0, 0.0, 2.0) if np.random.random() < 0.8 else None
            temp = Quantity(1.0, 0.0, 2.0) if np.random.random() < 0.8 else None
            hamil.t1 = t1
            hamil.t2star = t2star
            hamil.temp = temp
            if t1 is not None and t2star is not None and temp is not None:
                # Valid parameters should work
                hamil.get_jump_operators()
            else:
                # Invalid parameters should raise an exception
                with pytest.raises(ConfigurationException):
                    hamil.get_jump_operators()
