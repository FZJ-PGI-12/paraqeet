"""Test the Qubit model."""

import pytest
import numpy as np

from cthree.Quantity import Quantity
from cthree.model.GeneratorDrive import GeneratorDrive
from cthree.model.Qubit import Qubit
from cthree.signal.SimpleGenerator import CosGenerator
from cthree.signal.Device import CosToneErf


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
    return CosToneErf()


@pytest.fixture
def gen(tone):
    """Return a sinusoidal tone generator object."""
    gen = CosGenerator(devices=[tone])
    return gen


@pytest.fixture
def ham(gen):
    """Return a qubit."""
    drive = GeneratorDrive(gen, isLongitudinal=False)
    return Qubit(Quantity(FREQ, 0.8 * FREQ, 1.2 * FREQ), drives=[drive])


def test_getMatrix(ham, time_samples):
    """Test the getMatrix method."""
    hams = ham.getMatrix(time_samples)
    assert hams.shape == time_samples.shape + (2, 2)


def test_gradient(gen, ham, time_samples):
    """Test the gradients of the system.

    The Hamiltonian should have all derivatives of the drive
    plus the derivative w.r.t. the qubit frequency.

    """
    grads = gen.generateSignalGradient(time_samples)
    ham.setOptimisableParameters(ham.getParameters())
    hamGrads = ham.gradient(time_samples)

    assert hamGrads.shape == (grads.shape[0], grads.shape[1] + 1, 2, 2)
