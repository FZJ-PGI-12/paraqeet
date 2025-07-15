"""Test the signal gradient functions."""

import pytest
import numpy as np
import jax.numpy as jnp

from paraqeet.quantity import Array
from paraqeet.signal.envelopes import FlatTopGaussianEnvelope
from test.dummy_device import FlatTopGaussianEnvelopeAD

time = jnp.linspace(0, 10e-6, 100)


@pytest.fixture
def tone():
    """Tone with analytic gradient.

    Returns
    -------
    paraqeet.signal.envelopes.Envelope
        A Flat top Gaussian Envelope
    """
    return FlatTopGaussianEnvelope()


@pytest.fixture
def toneAD():
    """Tone with AutoDiff gradients.

    Returns
    -------
    paraqeet.signal.envelopes.Envelope
        A Flat top Gaussian Envelope without gradients defined
    """
    return FlatTopGaussianEnvelopeAD()


def random_entries_from_list(elements: Array, num: int = 0) -> Array:
    """Get random entries from an array of elements.

    Parameters
    ----------
    elements : jax.numpy.array
        Array of elements to choose from.
    num : int
        Number of random entries asked for.

    Returns
    -------
    jax.numpy.array
        Returns random entries selected from a list of elements.

    """
    if num is None:
        num = np.random.randint(0, len(elements))
    indices = np.random.choice(len(elements), num, replace=False)
    return np.array(elements)[indices]


def test_values(tone, toneAD):
    """Test values from the AD signal."""
    assert toneAD.compute_output(time) == pytest.approx(tone.compute_output(time), rel=1e-6)


def test_gradients(tone, toneAD):
    """Test generation of analytic and AD signal gradients one at a time."""
    grad_tone = []
    grad_toneAD = []

    # Pick a random number of parameters from the tones
    numParams = np.random.randint(0, len(tone.get_parameters()))
    for t in time:
        tone.set_optimisable_parameters(random_entries_from_list(tone.get_parameters(), numParams))
        toneAD.set_optimisable_parameters(random_entries_from_list(toneAD.get_parameters(), numParams))

        grad_toneAD.append(toneAD.compute_gradient(t))
        grad_tone.append(tone.compute_gradient(t))

    assert jnp.array(grad_toneAD) == pytest.approx(jnp.array(grad_tone), rel=1e-6)


def test_gradients_vectorized(tone, toneAD):
    """Test vectorized generation of analytic and AD signal gradients."""
    numParams = np.random.randint(0, len(tone.get_parameters()))

    toneAD.set_optimisable_parameters(random_entries_from_list(toneAD.get_parameters(), numParams))
    grad_toneAD = toneAD.compute_gradient(time)
    tone.set_optimisable_parameters(random_entries_from_list(tone.get_parameters(), numParams))
    grad_tone = tone.compute_gradient(time)
    assert jnp.array(grad_toneAD) == pytest.approx(jnp.array(grad_tone), rel=1e-6)


def test_gradients_shape(tone):
    """Test the signal gradient shape."""
    tone.set_optimisable_parameters(random_entries_from_list(tone.get_parameters()))
    grads = tone.compute_gradient(time)
    assert grads.shape[0] == time.shape[0]
