"""Test the functionality of the generation of the DRAG correted signal."""

import numpy as np
import pytest

from cthree.signal.Envelopes import (
    GaussEnvelope,
    ZeroEnvelope,
)
from cthree.signal.SimpleGenerator import CosGenerator
from cthree.signal.Waveform import DRAGMixer

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
    return CosGenerator(envelopes=[drag_tone])


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
    return CosGenerator(envelopes=[drag_tone])


@pytest.fixture
def genMultipleTones():
    """Return a multiple tone DRAG generator."""
    tone1 = GaussEnvelope()
    params1 = tone1.getParameters()
    drag_tone1 = DRAGMixer(tone1)

    tone2 = GaussEnvelope()
    params2 = tone2.getParameters()
    drag_tone2 = DRAGMixer(tone2)
    return (
        CosGenerator(envelopes=[drag_tone1, drag_tone2]),
        np.concatenate((params1, params2)),
    )


def test_constant_env(zeroGen, time_samples):
    """Test the values of a DRAG signal using a constant envelope."""
    assert np.all(
        zeroGen.generateSignal(time_samples) == np.zeros_like(time_samples)
    )


def test_gen(gen, time_samples) -> None:
    """Computes a sample signal and checks vectorized generation."""
    sig = gen.generateSignal(time_samples)
    assert len(sig) == LEN_SIG


def test_getParameters(genMultipleTones):
    """Test if the expected amount of parameters is present.

    If multiple tones define the total envelope, the signal needs to have the
    sum of all parameters of the envelope tones plus 4 parameters.

    """
    gen, all_params = genMultipleTones
    params = gen.getParameters()
    assert len(params) == len(all_params) + 4


def test_gradient_shape(gen, time_samples):
    """Test the length of the signal gradient."""
    gen.setOptimisableParameters(gen.getParameters())
    grads = gen.generateSignalGradient(time_samples)
    assert grads.shape[0] == time_samples.shape[0]
