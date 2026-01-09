"""Test the Generator Drive model."""

import numpy as np
import pytest

from paraqeet.model.drive_operator import DriveOperator
from paraqeet.signal.envelopes import FlatTopGaussianEnvelope
from paraqeet.signal.iq_mixer import IQMixer

LEN_SIG = 101


@pytest.fixture
def time_samples():
    """Return generated time samples from the given signal length.

    The given signal length `LEN_SIG` is a module level global variable.

    """
    return np.linspace(0, 10e-9, LEN_SIG)


@pytest.fixture
def tone():
    """Generate a sinusoidal tone."""
    tone = FlatTopGaussianEnvelope()
    tone.set_optimizable_parameters(tone.get_parameters())
    return tone


@pytest.fixture
def gen(tone):
    """Return a sinusoidal tone generator object."""
    gen = IQMixer(envelopes=[tone])
    return gen


@pytest.fixture
def drive(gen):
    """Return a generator drive object."""
    drive = DriveOperator(gen, is_longitudinal=False)
    return drive


def test_drive_get_value(drive, time_samples):
    """Test the drive getMatrix method."""
    dim = np.random.randint(2, 10)
    annihilation_op = np.sqrt(np.diag(np.arange(1, dim, dtype=np.float64), k=1))
    driveMatrices = drive.get_value(annihilation_op, time_samples)
    assert driveMatrices.shape == time_samples.shape + (dim, dim)


def test_drive_gradient(tone, drive, time_samples):
    """Test the drive gradient."""
    dim = np.random.randint(2, 10)
    annihilation_op = np.sqrt(np.diag(np.arange(1, dim, dtype=np.float64), k=1))
    grads = drive.get_gradient(annihilation_op, time_samples)
    tone_params = tone.get_parameters()
    assert grads.shape == time_samples.shape + (len(tone_params), dim, dim)


def test_has_parameters(drive):
    assert drive.get_parameters() is not None
    assert len(drive.get_parameters()) >= 0
    assert drive.generator is not None
