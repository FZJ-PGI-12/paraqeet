"""Test the Generator Drive model."""

import numpy as np
import pytest

from paraqeet.model.drive import DriveGenerator
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
    """Return a drive"""
    dim = np.random.randint(2, 10)
    annihilation_op = np.sqrt(np.diag(np.arange(1, dim, dtype=np.float64), k=1))
    drive_op = annihilation_op + annihilation_op.conj().T
    drive = DriveGenerator(drive_op, gen)
    return drive


def test_drive_get_value(drive, time_samples):
    """Test the drive get_value method."""
    drive_matrices = drive.get_value(time_samples)
    dim = drive.drive_op.shape[0]
    assert drive_matrices.shape == time_samples.shape + (dim, dim)


def test_drive_gradient(tone, drive, time_samples):
    """Test the drive gradient."""
    grads = drive.get_gradient(time_samples)
    tone_params = tone.get_parameters()
    dim = drive.drive_op.shape[0]
    assert grads.shape == time_samples.shape + (len(tone_params), dim, dim)


def test_has_parameters(drive):
    assert drive.get_parameters() is not None
    assert len(drive.get_parameters()) >= 0
    assert drive.generator is not None
