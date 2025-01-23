"""Test the Resonator model."""

import pytest
import numpy as np

from cthree.Quantity import Quantity
from cthree.model.DriveOperator import DriveOperator
from cthree.model.Resonator import Resonator
from cthree.signal.IQMixer import IQMixer
from cthree.signal.Envelopes import FlatTopGaussianEnvelope


FREQ = 4.8e9 * 2 * np.pi
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
    """Return a resonator object."""

    def _method(dimension):
        drive = DriveOperator(gen, isLongitudinal=False)
        return Resonator(
            dimension=dimension,
            frequency=Quantity(FREQ, 0.8 * FREQ, 1.2 * FREQ),
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
    plus the derivative w.r.t. the resonator frequency.

    """
    for dim in np.arange(1, 10):
        H = hamiltonian(dim)
        H.set_optimisable_parameters(H.get_parameters())
        grads = gen.generate_signal_gradient(time_samples)
        hamGrads = H.gradient(time_samples)
        assert hamGrads.shape == (grads.shape[0], grads.shape[1] + 1, dim, dim)
