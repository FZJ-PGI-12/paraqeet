"""Test the Generator Drive model."""

import pytest
import numpy as np

from cthree.signal.Device import CosToneErf
from cthree.signal.SimpleGenerator import CosGenerator
from cthree.model.GeneratorDrive import GeneratorDrive


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
    tone = CosToneErf()
    tone.setOptimisableParameters(tone.getParameters())
    return tone


@pytest.fixture
def gen(tone):
    """Return a sinusoidal tone generator object."""
    gen = CosGenerator(devices=[tone])
    return gen


@pytest.fixture
def drive(gen):
    """Return a generator drive object."""
    drive = GeneratorDrive(gen, isLongitudinal=False)
    return drive


def test_drive_getMatrix(drive, time_samples):
    """Test the drive getMatrix method."""
    dim = np.random.randint(2, 10)
    annihilationOp = np.sqrt(np.diag(np.arange(1, dim, dtype=np.float64), k=1))
    driveMatrices = drive.getMatrix(annihilationOp, time_samples)
    assert driveMatrices.shape == time_samples.shape + (dim, dim)


def test_drive_gradient(tone, drive, time_samples):
    """Test the drive gradient."""
    dim = np.random.randint(2, 10)
    annihilationOp = np.sqrt(np.diag(np.arange(1, dim, dtype=np.float64), k=1))
    grads = drive.gradient(annihilationOp, time_samples)
    toneParams = tone.getParameters()
    assert grads.shape == time_samples.shape + (len(toneParams), dim, dim)
