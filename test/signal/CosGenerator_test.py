"""Testing the cosine signal generator functions."""

import numpy as np
import pytest

from cthree.signal.Envelopes import ConstantEnvelope, ZeroEnvelope
from cthree.signal.IQMixer import IQMixer

LEN_SIG = 1001


@pytest.fixture
def time_samples():
    """Generate time samples from the given signal length.

    The given signal length `LEN_SIG` is a module level global variable.

    """
    return np.linspace(0, 10e-9, LEN_SIG)


@pytest.fixture
def gen():
    """Return cosine signal generator object."""
    tone = ConstantEnvelope()
    tone.set_optimisable_parameters(tone.get_parameters())
    return IQMixer(envelopes=[tone])


@pytest.fixture
def zeroGen():
    """Return a zero tone cosine generator object."""
    tone = ZeroEnvelope()
    return IQMixer(envelopes=[tone])


@pytest.fixture
def genMultipleTones():
    """Return a multiple tone cosine generator."""
    tone1 = ConstantEnvelope()
    params1 = tone1.get_parameters()

    tone2 = ConstantEnvelope()
    params2 = tone2.get_parameters()
    return IQMixer(envelopes=[tone1, tone2]), np.concatenate((params1, params2))


def test_gen(gen, time_samples) -> None:
    """Computes a sample signal and checks vectorized generation."""
    sig = gen.generate_signal(time_samples)
    assert len(sig) == LEN_SIG


def test_zeroTone(zeroGen, time_samples) -> None:
    """Test generation of zeroTone signal."""
    sig = zeroGen.generate_signal(time_samples)
    assert len(sig) == LEN_SIG
    assert np.all(sig == 0)


def test_getParamters(genMultipleTones):
    """Test the get parameters function.

    First four parameters are the tone parameters and last two in 'params' are
    added by the generator, i.e., phase and LO frequency.

    """
    gen, all_params = genMultipleTones
    params = gen.get_parameters()[:-2]
    assert np.all(params == all_params)


def test_gradientOneTime(gen):
    """Test the generate signal gradient one time function."""
    print(gen.get_parameters())
    gen.set_optimisable_parameters(gen.get_parameters())
    grads = gen.generate_signal_gradient_one_time(0)
    assert grads.shape == (len(gen.get_parameters()),)


def test_gradient_shape(gen, time_samples):
    """Test the generate signal gradient function."""
    gen.set_optimisable_parameters(gen.get_parameters())
    grads = gen.generate_signal_gradient(time_samples)
    assert grads.shape == (time_samples.shape[0], len(gen.get_parameters()))
