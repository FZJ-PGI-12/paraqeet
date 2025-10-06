"""Test the signal gradient functions."""

import pytest
import numpy as np
import jax.numpy as jnp

from paraqeet.quantity import Array
from paraqeet.signal.envelopes import FlatTopGaussianEnvelope
from tests.dummy_device import FlatTopGaussianEnvelopeAD

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
def tone_ad():
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
    elements: Array
        Array of elements to choose from.
    num: int
        Number of random entries asked for.

    Returns
    -------
    Array
        Returns random entries selected from a list of elements.

    """
    if num is None:
        num = np.random.randint(0, len(elements))
    indices = np.random.choice(len(elements), num, replace=False)
    return np.array(elements)[indices]


def test_values(tone, tone_ad):
    """Test values from the AD signal."""
    assert tone_ad.compute_output(time) == pytest.approx(tone.compute_output(time), rel=1e-6)


def test_gradients(tone, tone_ad):
    """Test generation of analytic and AD signal gradients one at a time."""
    grad_tone = []
    grad_tone_ad = []

    # Pick a random number of parameters from the tones
    num_params = np.random.randint(0, len(tone.get_parameters()))
    for t in time:
        tone.set_optimizable_parameters(random_entries_from_list(tone.get_parameters(), num_params))
        tone_ad.set_optimizable_parameters(random_entries_from_list(tone_ad.get_parameters(), num_params))

        grad_tone_ad.append(tone_ad.compute_gradient(t))
        grad_tone.append(tone.compute_gradient(t))

    assert jnp.array(grad_tone_ad) == pytest.approx(jnp.array(grad_tone), rel=1e-6)


def test_gradients_vectorized(tone, tone_ad):
    """Test vectorized generation of analytic and AD signal gradients."""
    num_params = np.random.randint(0, len(tone.get_parameters()))

    tone_ad.set_optimizable_parameters(random_entries_from_list(tone_ad.get_parameters(), num_params))
    grad_tone_ad = tone_ad.compute_gradient(time)
    tone.set_optimizable_parameters(random_entries_from_list(tone.get_parameters(), num_params))
    grad_tone = tone.compute_gradient(time)
    assert jnp.array(grad_tone_ad) == pytest.approx(jnp.array(grad_tone), rel=1e-6)


def test_gradients_shape(tone):
    """Test the signal gradient shape."""
    tone.set_optimizable_parameters(random_entries_from_list(tone.get_parameters()))
    grads = tone.compute_gradient(time)
    assert grads.shape[0] == time.shape[0]
