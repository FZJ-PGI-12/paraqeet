from typing import List

import pytest
import numpy as np
import jax.numpy as jnp

from cthree.signal.Device import CosToneAD, CosToneErfAD
from cthree.signal.Device import CosTone, CosToneErf

costone = CosTone()
costoneerf = CosToneErf()
costoneAD = CosToneAD()
costoneerfAD = CosToneErfAD()
time = jnp.linspace(0, 10e-6, 100)


def randomEntriesFromList(elements: jnp.array, num: int = None) -> jnp.array:
    if num is None:
        num = np.random.randint(0, len(elements))
    indices = np.random.choice(len(elements), num, replace=False)
    return np.array(elements)[indices]


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

    # Pick a random number of parameters from the tones
    numParams = np.random.randint(0, len(tone.getParameters()))
    for t in time:
        tone.setOptimisableParameters(randomEntriesFromList(tone.getParameters(), numParams))
        toneAD.setOptimisableParameters(randomEntriesFromList(toneAD.getParameters(), numParams))

        grad_toneAD.append(toneAD.computeGradient(t))
        grad_tone.append(tone.computeGradient(t))

    assert jnp.array(grad_toneAD) == pytest.approx(jnp.array(grad_tone), rel=1e-6)


@pytest.mark.parametrize("tone, toneAD", [(costone, costoneAD), (costoneerf, costoneerfAD)])
def test_gradients_vectorized(tone, toneAD):
    """
    Test vectorized generation of analytic and AD signal gradients
    """
    numParams = np.random.randint(0, len(tone.getParameters()))

    toneAD.setOptimisableParameters(randomEntriesFromList(toneAD.getParameters(), numParams))
    grad_toneAD = toneAD.computeGradient(time)
    tone.setOptimisableParameters(randomEntriesFromList(tone.getParameters(), numParams))
    grad_tone = tone.computeGradient(time)
    assert jnp.array(grad_toneAD) == pytest.approx(jnp.array(grad_tone), rel=1e-6)


@pytest.mark.parametrize("tone", [costone, costoneAD, costoneerf, costoneerfAD])
def test_gradients_shape(tone):
    tone.setOptimisableParameters(randomEntriesFromList(tone.getParameters()))
    grads = tone.computeGradient(time)
    assert grads.shape[0] == time.shape[0]
