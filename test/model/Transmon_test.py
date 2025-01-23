"""Test the Transmon object."""

import pytest
import numpy as np

from cthree.Quantity import Quantity
from cthree.model.DriveOperator import DriveOperator
from cthree.model.Transmon import Transmon
from cthree.signal.IQMixer import IQMixer
from cthree.signal.Envelopes import FlatTopGaussianEnvelope


FREQ = 4.8e9 * 2 * np.pi
ANHARMONICITY = -200e6 * 2 * np.pi
LEN_SIG = 1001


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


def test_getMatrix(hamiltonian, time_samples):
    """Test the getMatrix method."""
    for dim in np.arange(1, 10):
        H = hamiltonian(dim)
        hams = H.getMatrix(time_samples)
        assert hams.shape == time_samples.shape + (dim, dim)


def test_gradient(gen, hamiltonian, time_samples):
    """Test the gradients of the system.

    The Hamiltonian should have all derivatives of the drive
    plus the derivative w.r.t. the frequency and anharmonicity.

    """
    for dim in np.arange(1, 10):
        H = hamiltonian(dim)
        H.set_optimisable_parameters(H.get_parameters())
        grads = gen.generateSignalGradient(time_samples)
        hamGrads = H.gradient(time_samples)
        assert hamGrads.shape == (grads.shape[0], grads.shape[1] + 2, dim, dim)


def test_getDriveMatrix(hamiltonian, time_samples):
    """Test the getDriveMatrix method of the Hamiltonian."""
    dim = np.random.randint(2, 10)
    annihilationOp = np.sqrt(np.diag(np.arange(1, dim, dtype=np.float64), k=1))
    H = hamiltonian(dim)
    driveMatrix = H._getDriveMatrix(annihilationOp, time_samples)
    assert driveMatrix.shape == time_samples.shape + (dim, dim)


def test_getDriveGradients(gen, hamiltonian, time_samples):
    """Test the drive gradients of the Hamiltonian."""
    dim = np.random.randint(2, 10)
    annihilationOp = np.sqrt(np.diag(np.arange(1, dim, dtype=np.float64), k=1))
    H = hamiltonian(dim)
    grads = gen.generateSignalGradient(time_samples)
    driveGradients = H._getDriveGradients(annihilationOp, time_samples)
    assert driveGradients.shape == (grads.shape[0], grads.shape[1], dim, dim)
