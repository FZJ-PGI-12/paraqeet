import pytest
import jax.numpy as np

from cthree.signal.Device import CosToneAD, CosToneErfAD
from cthree.signal.Device import CosTone, CosToneErf

costone = CosTone()
costoneerf = CosToneErf()
costoneAD = CosToneAD()
costoneerfAD = CosToneErfAD()
time = np.linspace(0, 10e-6, 100)


@pytest.mark.parametrize("tone, toneAD", [(costone, costoneAD), (costoneerf, costoneerfAD)])
def test_values(tone, toneAD):
    assert toneAD.computeOutput(time) == pytest.approx(
        tone.computeOutput(time), rel=1e-6
    )

@pytest.mark.parametrize("tone, toneAD", [(costone, costoneAD), (costoneerf, costoneerfAD)])
def test_gradients(tone, toneAD):
    """
    Test generation of analytic and AD signal gradients one at a time
    """
    grad_tone = []
    grad_toneAD = []

    for t in time:
        grad_toneAD.append(toneAD.computeGradient(t))
        grad_tone.append(tone.computeGradient(t))

    assert np.array(grad_toneAD) == pytest.approx(np.array(grad_tone), rel=1e-6)

@pytest.mark.parametrize("tone, toneAD", [(costone, costoneAD), (costoneerf, costoneerfAD)])
def test_gradients_vectorized(tone, toneAD):
    """
    Test vectorized generation of analytic and AD signal gradients
    """
    grad_toneAD = toneAD.computeGradient(time)
    grad_tone = tone.computeGradient(time)
    assert np.array(grad_toneAD) == pytest.approx(np.array(grad_tone), rel=1e-6)

@pytest.mark.parametrize("tone", [costone, costoneAD, costoneerf, costoneerfAD])
def test_gradients_shape(tone):
    grads = tone.computeGradient(time)
    params = tone.getParameters()
    assert grads.shape[0] == time.shape[0]
