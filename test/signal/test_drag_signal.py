"""Test the functionality of the generation of the DRAG correted signal."""

from abc import ABC

import numpy as np
import pytest
from jax import numpy as jnp

from cthree.signal.DRAGGenerator import DRAGGenerator
from cthree.signal.Device import GaussTone, ZeroTone, Device

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
    tone = GaussTone()
    return DRAGGenerator(devices=[tone])


@pytest.fixture
def zero():
    """Return a zero signal device."""
    return ZeroTone()


@pytest.fixture
def gauss():
    """Return a gaussian signal device."""
    return GaussTone()


@pytest.fixture
def zeroGen():
    """Return a zero tone DRAG generator object."""
    tone = ZeroTone()
    return DRAGGenerator(devices=[tone])


@pytest.fixture
def genMultipleTones():
    """Return a multiple tone DRAG generator."""
    tone1 = GaussTone()
    params1 = tone1.getParameters()

    tone2 = GaussTone()
    params2 = tone2.getParameters()
    return (
        DRAGGenerator(devices=[tone1, tone2]),
        np.concatenate((params1, params2)),
    )


def test_constant_env(zeroGen, time_samples):
    """Test the values of a DRAG signal using a constant envelope."""
    assert np.all(
        zeroGen.generateSignal(time_samples) == np.zeros_like(time_samples)
    )


def test_env_grad_equality(zero, gauss, time_samples):
    """Test time gradients.

    Test if the autograd time gradients match the ones calculated by hand
    for a gaussian signal envelope.

    """

    class GaussTonewithoutgrad(Device, ABC):
        def __init__(self) -> None:
            self.__amplitude = gauss.getParameters()[0]
            self.__duration = gauss.getParameters()[1]

        def computeEnvelope(self, t: np.ndarray):
            dur = self.__duration.getValue()
            sigma = dur / 6
            env = self.__amplitude.getValue()
            env *= jnp.exp(-(1 / 2) * (t - dur / 2) ** 2 / sigma**2)
            return jnp.squeeze(env)

    class GaussTonewithoutgradandenv(Device, ABC):
        def __init__(self) -> None:
            self.__amplitude = gauss.getParameters()[0]
            self.__duration = gauss.getParameters()[1]

    autoderiv = GaussTonewithoutgrad().computeEnvelopeTimeGradient(time_samples)
    deriv = gauss.computeEnvelopeTimeGradient(time_samples)

    with pytest.raises(NotImplementedError):
        GaussTonewithoutgradandenv().computeEnvelopeTimeGradient(time_samples)
    assert len(autoderiv) == len(deriv) == LEN_SIG
    assert np.all(abs((autoderiv - deriv) / deriv) < 1.5 * 10 ** (-8))
    assert np.all(
        abs(
            (
                gauss.computeEnvelope(time_samples)
                - gauss.computeOutput(time_samples)
            )
            / gauss.computeOutput(time_samples)
        )
        < 1.5 * 10 ** (-8)
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
