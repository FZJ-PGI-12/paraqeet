"""Test the Transmon object."""

import pytest
import numpy as np

from cthree.model.open_system import OpenSystem
from cthree.propagation.scipy_expm import ScipyExpm
from cthree.propagation.vern7 import Vern7
from cthree.quantity import Quantity
from cthree.model.drive_operator import DriveOperator
from cthree.model.transmon import Transmon
from cthree.signal.iq_mixer import IQMixer
from cthree.signal.envelopes import FlatTopGaussianEnvelope, ZeroEnvelope

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

    def _method(dimension):
        drive = DriveOperator(gen, isLongitudinal=False)
        drive.set_optimisable_parameters(drive.get_parameters())
        return Transmon(
            dimension=dimension,
            frequency=Quantity(FREQ, 0.8 * FREQ, 1.2 * FREQ),
            anharmonicity=Quantity(ANHARMONICITY, 1.2 * ANHARMONICITY, 0.8 * ANHARMONICITY),
            drives=[drive],
        )

    return _method


@pytest.fixture
def openTransmon():
    """Return an open model for the resonator."""

    tone = ZeroEnvelope()
    generator = IQMixer(envelopes=[tone])
    drive = DriveOperator(generator, isLongitudinal=False)
    resonator = Transmon(
        frequency=Quantity(FREQ, 0.8 * FREQ, 1.2 * FREQ),
        anharmonicity=Quantity(ANHARMONICITY, 1.2 * ANHARMONICITY, 0.8 * ANHARMONICITY),
        drives=[drive],
        dimension=DIMS,
    )
    resonator.t1 = T1
    resonator.temp = TEMP
    resonator.t2star = T2STAR
    model = OpenSystem(resonator)

    return model


@pytest.fixture
def expm(openTransmon):
    init = np.zeros((DIMS, 1), dtype=np.complex128)
    init[DIMS - 1][0] = 1  # Fully excited state
    init_dm = np.matmul(init, init.T)

    prop = ScipyExpm(openTransmon, res=100e9)
    prop.set_initial_state(init_dm)
    return prop


@pytest.fixture
def ode(openTransmon):
    init = np.zeros((DIMS, 1), dtype=np.complex128)
    init[DIMS - 1][0] = 1  # Fully excited state
    init_dm = np.matmul(init, init.T)

    openTransmon.ode_propagation = True
    prop = Vern7(openTransmon, res=100e9)
    prop.set_initial_state(init_dm)
    return prop


def test_get_matrix(hamiltonian, time_samples):
    """Test the getMatrix method."""
    for dim in np.arange(1, 10):
        H = hamiltonian(dim)
        hams = H.get_matrix(time_samples)
        assert hams.shape == time_samples.shape + (dim, dim)


def test_gradient(gen, hamiltonian, time_samples):
    """Test the gradients of the system.

    The Hamiltonian should have all derivatives of the drive
    plus the derivative w.r.t. the frequency and anharmonicity.

    """
    for dim in np.arange(1, 10):
        H = hamiltonian(dim)
        H.set_optimisable_parameters(H.get_parameters())
        grads = gen.generate_signal_gradient(time_samples)
        hamGrads = H.gradient(time_samples)
        assert hamGrads.shape == (grads.shape[0], grads.shape[1] + 2, dim, dim)


def test_get_drive_matrix(hamiltonian, time_samples):
    """Test the getDriveMatrix method of the Hamiltonian."""
    dim = np.random.randint(2, 10)
    annihilationOp = np.sqrt(np.diag(np.arange(1, dim, dtype=np.float64), k=1))
    H = hamiltonian(dim)
    driveMatrix = H._get_drive_matrix(annihilationOp, time_samples)
    assert driveMatrix.shape == time_samples.shape + (dim, dim)


def test_get_drive_gradients(gen, hamiltonian, time_samples):
    """Test the drive gradients of the Hamiltonian."""
    dim = np.random.randint(2, 10)
    annihilationOp = np.sqrt(np.diag(np.arange(1, dim, dtype=np.float64), k=1))
    H = hamiltonian(dim)
    grads = gen.generate_signal_gradient(time_samples)
    driveGradients = H._get_drive_gradients(annihilationOp, time_samples)
    assert driveGradients.shape == (grads.shape[0], grads.shape[1], dim, dim)


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
