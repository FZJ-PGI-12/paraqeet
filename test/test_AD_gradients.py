import pytest
import jax.numpy as np

from cthree.signal.CosToneAD import CosToneAD
from cthree.signal.Device import CosTone


tone = CosTone()
time = np.linspace(0, 10e-6, 100)


@pytest.fixture
def tone_AD():
    return CosToneAD()


def test_values(tone_AD):
    assert tone_AD.computeOutput(time) == pytest.approx(
        tone.computeOutput(time), rel=1e-4
    )


def test_gradients(tone_AD):
    grad_tone_AD = tone_AD.computeGradient(time)
    grad_tone = tone.computeGradient(time)
    assert np.array(grad_tone_AD) == pytest.approx(np.array(grad_tone), rel=1e-4)
