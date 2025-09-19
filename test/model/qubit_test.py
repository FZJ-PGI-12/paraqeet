"""Test the Qubit model."""

import pytest
import numpy as np

from paraqeet.exceptions import ConfigurationException
from paraqeet.quantity import Quantity
from paraqeet.model.drive_operator import DriveOperator
from paraqeet.model.qubit import Qubit
from paraqeet.signal.iq_mixer import IQMixer
from paraqeet.signal.envelopes import FlatTopGaussianEnvelope

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
def ham(gen):
    """Return a qubit."""
    drive = DriveOperator(gen, is_longitudinal=False)
    return Qubit(Quantity(FREQ, 0.8 * FREQ, 1.2 * FREQ), drives=[drive])


def test_get_matrix(ham, time_samples):
    """Test the getMatrix method."""
    hams = ham.get_matrix(time_samples)
    assert hams.shape == time_samples.shape + (2, 2)


def test_gradient(gen, ham, time_samples):
    """Test the gradients of the system.

    The Hamiltonian should have all derivatives of the drive
    plus the derivative w.r.t. the qubit frequency.

    """
    grads = gen.generate_signal_gradient(time_samples)
    ham.set_optimisable_parameters(ham.get_parameters())
    ham_grads = ham.gradient(time_samples)
    assert ham_grads.shape == (len(time_samples), grads.shape[1] + 1, 2, 2)

    ham.set_optimisable_parameters([ham.frequency])
    ham_grads = ham.gradient(time_samples)
    assert ham_grads.shape == (len(time_samples), 1, 2, 2)


def test_setters_and_getters(ham, random_quantity):
    frequency = random_quantity(1)
    ham.frequency = frequency
    assert ham.frequency == frequency


def test_needs_parameters_for_decay_rates(ham):
    for _ in range(10):
        t1 = Quantity(1.0, 0.0, 2.0) if np.random.random() < 0.8 else None
        t2star = Quantity(1.0, 0.0, 2.0) if np.random.random() < 0.8 else None
        temp = Quantity(1.0, 0.0, 2.0) if np.random.random() < 0.8 else None
        ham.t1 = t1
        ham.t2star = t2star
        ham.temp = temp
        if t1 is not None and t2star is not None and temp is not None:
            # Valid parameters should work
            ham.get_collapseops()
        else:
            # Invalid parameters should raise an exception
            with pytest.raises(ConfigurationException):
                ham.get_collapseops()
