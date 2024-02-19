import pytest
import numpy as np

from cthree.Quantity import Quantity
from cthree.model.GeneratorDrive import GeneratorDrive
from cthree.model.Qubit import Qubit
from cthree.signal.SimpleGenerator import CosGenerator
from cthree.signal.Device import CosToneErf


FREQ = 4.8e9 * 2 * np.pi
LEN_SIG = 1001


@pytest.fixture
def time_samples():
    return np.linspace(0, 10e-9, LEN_SIG)


@pytest.fixture
def tone():
    return CosToneErf()


@pytest.fixture
def gen(tone):
    gen = CosGenerator(devices=[tone])
    return gen


@pytest.fixture
def ham(gen):
    drive = GeneratorDrive(gen, isLongitudinal=False)
    return Qubit(Quantity(FREQ, 0.8*FREQ, 1.2*FREQ), drives=[drive])


def test_getMatrix(ham, time_samples):
    hams = ham.getMatrix(time_samples)
    assert hams.shape == time_samples.shape + (2, 2)


# The Hamiltonian should have all derivatives of the drive plus the derivative w.r.t. the qubit frequency
def test_gradient(gen, ham, time_samples):
    grads = gen.generateSignalGradient(time_samples)
    hamGrads = ham.gradient(time_samples)

    assert hamGrads.shape == (grads.shape[0], grads.shape[1] + 1, 2, 2)
