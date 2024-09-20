"""Testing the cosine signal generator functions."""

import numpy as np
import pytest

from cthree.signal.Device import CosTone
from cthree.signal.Device import ZeroTone
from cthree.signal.SimpleGenerator import CosGenerator

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
    tone = CosTone()
    tone.setOptimisableParameters(tone.getParameters())
    return CosGenerator(devices=[tone])


@pytest.fixture
def zeroGen():
    """Return a zero tone cosine generator object."""
    tone = ZeroTone()
    return CosGenerator(devices=[tone])


@pytest.fixture
def genMultipleTones():
    """Return a multiple tone cosine generator."""
    tone1 = CosTone()
    params1 = tone1.getParameters()

    tone2 = CosTone()
    params2 = tone2.getParameters()
    return CosGenerator(devices=[tone1, tone2]), np.concatenate(
        (params1, params2)
    )


def test_gen(gen, time_samples) -> None:
    """Computes a sample signal and checks vectorized generation."""
    sig = gen.generateSignal(time_samples)
    assert len(sig) == LEN_SIG


def test_zeroTone(zeroGen, time_samples) -> None:
    """Test generation of zeroTone signal."""
    sig = zeroGen.generateSignal(time_samples)
    assert len(sig) == LEN_SIG
    assert np.all(sig == 0)


def test_getParamters(genMultipleTones):
    """Test the get parameters function."""
    gen, all_params = genMultipleTones
    params = gen.getParameters()[:4]
    assert np.all(params == all_params)


def test_gradientOneTime(gen):
    """Test the generate signal gradient one time function."""
    gen.setOptimisableParameters(gen.getParameters())
    grads = gen.generateSignalGradientOneTime(0)
    assert grads.shape == (len(gen.getParameters()),)


def test_gradient_shape(gen, time_samples):
    """Test the generate signal gradient function."""
    gen.setOptimisableParameters(gen.getParameters())
    grads = gen.generateSignalGradient(time_samples)
    assert grads.shape == (time_samples.shape[0], len(gen.getParameters()))
