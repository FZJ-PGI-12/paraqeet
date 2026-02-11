import numpy as np
import pytest

from paraqeet.signal.waveform import LocalOscillator


@pytest.fixture
def local_oscillator() -> LocalOscillator:
    return LocalOscillator()


@pytest.fixture
def random_time_vector():
    return np.linspace(0.0, np.random.randint(1, 10) * np.random.rand(), np.random.randint(1, 10))


def test_getters_and_setters(local_oscillator, random_quantity):
    frequency = random_quantity(1, "Hz")
    local_oscillator.frequency = frequency
    assert local_oscillator.frequency == frequency

    assert local_oscillator.get_parameters() is not None
    assert len(local_oscillator.get_parameters()) >= 0
    assert frequency in local_oscillator.get_parameters()


def test_output_shapes(local_oscillator, random_time_vector):
    output = local_oscillator.get_value(random_time_vector)
    assert output.shape == (len(random_time_vector),)

    local_oscillator.set_optimizable_parameters(local_oscillator.get_parameters())
    _, gradient = local_oscillator.get_value_and_gradient(random_time_vector)
    assert gradient.shape == (len(random_time_vector), len(local_oscillator.get_parameters()))

    time_gradient = local_oscillator.get_time_gradient(random_time_vector)
    assert time_gradient.shape == (len(random_time_vector),)
