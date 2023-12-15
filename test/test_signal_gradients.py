import pytest
import jax.numpy as np

from cthree.signal.DeviceAD import CosToneAD, CosToneErfAD
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
        costone.computeOutput(time), rel=1e-4
    )
    assert costoneerfAD.computeOutput(time) == pytest.approx(
        costoneerf.computeOutput(time), rel=1e-4
    )


def test_gradients(costoneAD, costoneerfAD):
    grad_costone_AD = costoneAD.computeGradient(time)
    grad_costone = costone.computeGradient(time)

    grad_costoneerf_AD = costoneerfAD.computeGradient(time)
    grad_costoneerf = costoneerf.computeGradient(time)

    assert np.array(grad_costone_AD) == pytest.approx(np.array(grad_costone), rel=1e-4)
    assert np.array(grad_costoneerf_AD) == pytest.approx(
        np.array(grad_costoneerf), rel=1e-4
    )
