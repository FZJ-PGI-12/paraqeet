import pytest
import jax.numpy as np

from cthree.signal.Device import CosToneAD, CosToneErfAD
from cthree.signal.Device import CosTone, CosToneErf

costone = CosTone()
costoneerf = CosToneErf()
time = np.linspace(0, 10e-6, 100)


@pytest.fixture
def costoneAD():
    return CosToneAD()


@pytest.fixture
def costoneerfAD():
    return CosToneErfAD()


def test_values(costoneAD, costoneerfAD):
    assert costoneAD.computeOutput(time) == pytest.approx(
        costone.computeOutput(time), rel=1e-6
    )
    assert costoneerfAD.computeOutput(time) == pytest.approx(
        costoneerf.computeOutput(time), rel=1e-6
    )


def test_gradients(costoneAD, costoneerfAD):
    """
    Test generation of analytic and AD signal gradients one at a time
    """
    grad_costone = []
    grad_costone_AD = []

    grad_costoneerf = []
    grad_costoneerf_AD = []

    for t in time:
        grad_costone_AD.append(costoneAD.computeGradient(t))
        grad_costone.append(costone.computeGradient(t))

    grad_costoneerf_AD.append(costoneerfAD.computeGradient(t))
    grad_costoneerf.append(costoneerf.computeGradient(t))

    assert np.array(grad_costone_AD) == pytest.approx(np.array(grad_costone), rel=1e-6)
    assert np.array(grad_costoneerf_AD) == pytest.approx(
        np.array(grad_costoneerf), rel=1e-6
    )


def test_gradients_vectorized(costoneAD, costoneerfAD):
    """
    Test vectorized generation of analytic and AD signal gradients
    """
    grad_costone_AD = costoneAD.computeGradient(time)
    grad_costone = costone.computeGradient(time)

    grad_costoneerf_AD = costoneerfAD.computeGradient(time)
    grad_costoneerf = costoneerf.computeGradient(time)

    assert np.array(grad_costone_AD) == pytest.approx(np.array(grad_costone), rel=1e-6)
    assert np.array(grad_costoneerf_AD) == pytest.approx(
        np.array(grad_costoneerf), rel=1e-6
    )
