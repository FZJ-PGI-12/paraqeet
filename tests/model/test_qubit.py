"""Test the Qubit model."""

import numpy as np
import pytest

from paraqeet.exceptions import ConfigurationException
from paraqeet.model.drive import Drive
from paraqeet.model.qubit import Qubit, QubitHamiltonian
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import FlatTopGaussianEnvelope
from paraqeet.signal.iq_mixer import IQMixer

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
    pauli_x = np.array([[0.0, 1.0], [0.0, 1.0]])
    drive = Drive(pauli_x, gen)
    return QubitHamiltonian(Quantity(FREQ, 0.8 * FREQ, 1.2 * FREQ), drives=[drive])


def test_get_hamiltonian(ham, time_samples):
    """Test the get_hamiltonian method."""
    hams = ham.get_value(time_samples)
    assert hams.shape == time_samples.shape + (2, 2)


def test_gradient(gen, ham, time_samples):
    """Test the gradients of the system.

    The Hamiltonian should have all derivatives of the drive
    plus the derivative w.r.t. the qubit frequency.
    """

    grads = gen.get_gradient(time_samples)
    ham.set_optimizable_parameters(ham.get_parameters())
    ham_grads = ham.get_gradient(time_samples)
    assert ham_grads.shape == (
        len(time_samples),
        grads.shape[1] + 1,
        2,
        2,
    ), "The gradient shape should be (n_times, num_params_gen + 1, 2, 2)"

    ham.set_optimizable_parameters([ham.frequency])
    _, ham_grads = ham.get_value_and_gradient(time_samples)
    assert ham_grads.shape == (len(time_samples), 1, 2, 2), "The gradient shape should be (n_times, 1, 2, 2)"


def test_setters_and_getters(ham, random_quantity):
    frequency = random_quantity(1)
    ham.frequency = frequency
    assert ham.frequency == frequency


def test_needs_parameters_for_decay_rates(ham):
    for _ in range(10):
        t1 = Quantity(1.0, 0.0, 2.0) if np.random.random() < 0.8 else None
        t2star = Quantity(1.0, 0.0, 2.0) if np.random.random() < 0.8 else None
        temp = Quantity(1.0, 0.0, 2.0) if np.random.random() < 0.8 else None
        qubit = Qubit(
            hamiltonian=ham,
            t1=t1,
            temp=temp,
            t2star=t2star,
        )
        ham.t1 = t1
        ham.t2star = t2star
        ham.temp = temp
        if t1 is not None and t2star is not None and temp is not None:
            # Valid parameters should work
            qubit.get_jump_operators()
        else:
            # Invalid parameters should raise an exception
            with pytest.raises(ConfigurationException):
                qubit.get_jump_operators()
