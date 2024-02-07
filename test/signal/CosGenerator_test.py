import pytest
import numpy as np
from cthree.signal.SimpleGenerator import CosGenerator
from cthree.signal.Device import CosTone
from cthree.signal.Device import ZeroTone

LEN_SIG = 1001


@pytest.fixture
def time_samples():
    return np.linspace(0, 10e-9, LEN_SIG)


@pytest.fixture
def gen():
    tone = CosTone()
    return CosGenerator(devices=[tone])


@pytest.fixture
def zeroGen():
    tone = ZeroTone()
    return CosGenerator(devices=[tone])


@pytest.fixture
def genMultipleTones():
    tone1 = CosTone()
    params1 = tone1.getParameters()

    tone2 = CosTone()
    params2 = tone2.getParameters()
    return CosGenerator(devices=[tone1, tone2]), np.concatenate((params1, params2))


def test_gen(gen, time_samples) -> None:
    """
    Computes a sample signal and checks vectorized generation.
    """
    sig= gen.generateSignal(time_samples)
    assert len(sig) == LEN_SIG


def test_zeroTone(zeroGen, time_samples) -> None:
    """
    Test generation of zeroTone signal
    """
    sig = zeroGen.generateSignal(time_samples)
    assert len(sig) == LEN_SIG
    assert np.all(sig == 0)


def test_getParamters(genMultipleTones):
    gen, all_params = genMultipleTones
    params = gen.getParameters()
    assert np.all(params == all_params)


def test_gradient_shape(gen, time_samples):
    grads = gen.generateSignalGradient(time_samples)
    assert grads.shape[0] == time_samples.shape[0]