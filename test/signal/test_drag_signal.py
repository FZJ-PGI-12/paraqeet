"""Test the functionality of the generation of the DRAG correted signal."""

import numpy as np
import pytest

from paraqeet.signal.envelopes import (
    FlatTopGaussianEnvelope,
    GaussEnvelope,
    ZeroEnvelope,
)
from paraqeet.signal.iq_mixer import IQMixer
from paraqeet.signal.waveform import DRAGMixer

LEN_SIG = 1001


@pytest.fixture
def time_samples():
    """Generate time samples from the given signal length.

    The given signal length `LEN_SIG` is a module level global variable.

    """
    return np.linspace(0, 10e-9, LEN_SIG)


@pytest.fixture
def gen():
    """Return DRAG signal generator object."""
    tone = GaussEnvelope()
    drag_tone = DRAGMixer(tone)
    return IQMixer(envelopes=[drag_tone])


@pytest.fixture
def zero():
    """Return a zero signal device."""
    return ZeroEnvelope()


@pytest.fixture
def gauss():
    """Return a gaussian signal device."""
    return GaussEnvelope()


@pytest.fixture
def zeroGen():
    """Return a zero tone DRAG generator object."""
    tone = ZeroEnvelope()
    drag_tone = DRAGMixer(tone)
    return IQMixer(envelopes=[drag_tone])


@pytest.fixture
def flattop():
    """Return a FlatTop signal with DRAG."""
    tone = FlatTopGaussianEnvelope()
    drag_tone = DRAGMixer(tone)
    return IQMixer(envelopes=[drag_tone])


@pytest.fixture
def gen_multiple_tones():
    """Return a multiple tone DRAG generator."""
    tone1 = GaussEnvelope()
    params1 = tone1.get_parameters()
    drag_tone1 = DRAGMixer(tone1)

    tone2 = GaussEnvelope()
    params2 = tone2.get_parameters()
    drag_tone2 = DRAGMixer(tone2)
    return (
        IQMixer(envelopes=[drag_tone1, drag_tone2]),
        np.concatenate((params1, params2)),
    )


def test_constant_env(zeroGen, time_samples):
    """Test the values of a DRAG signal using a constant envelope."""
    assert np.all(zeroGen.generate_signal(time_samples) == np.zeros_like(time_samples))


def test_gen(gen, time_samples) -> None:
    """Computes a sample signal and checks vectorized generation."""
    sig = gen.generate_signal(time_samples)
    assert len(sig) == LEN_SIG


def test_getParameters(gen_multiple_tones):
    """Test if the expected amount of parameters is present.

    If multiple tones define the total envelope, the signal needs to have the
    sum of all parameters of the envelope tones plus 4 parameters.

    """
    gen, all_params = gen_multiple_tones
    params = gen.get_parameters()
    assert len(params) == len(all_params) + 4


def test_gradient_shape(gen, time_samples):
    """Test the length of the signal gradient."""
    gen.set_optimisable_parameters(gen.get_parameters())
    grads = gen.generate_signal_gradient(time_samples)
    assert grads.shape[0] == time_samples.shape[0]


def test_gradient_flattop(flattop, time_samples):
    """Test the length of the gradient of flattop signal."""
    flattop.set_optimisable_parameters(flattop.get_parameters())
    grads = flattop.generate_signal_gradient(time_samples)
    assert grads.shape[0] == time_samples.shape[0]
