import pytest
import numpy as np

from cthree.Quantity import Quantity
from cthree.model.GeneratorDrive import GeneratorDrive
from cthree.model.Resonator import Resonator
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
def hamiltonian(gen):
    def _method(dimension):
        drive = GeneratorDrive(gen, isLongitudinal=False)
        return Resonator(
            dimension=dimension,
            frequency=Quantity(FREQ, 0.8*FREQ, 1.2*FREQ),
            drives=[drive]
        )
    return _method


def test_getMatrix(hamiltonian, time_samples):
    for dim in np.arange(1, 10):
        H = hamiltonian(dim)
        hams = H.getMatrix(time_samples)
        assert hams.shape == time_samples.shape + (dim, dim)


# The Hamiltonian should have all derivatives of the drive plus the derivative w.r.t. the resonator frequency
def test_gradient(gen, hamiltonian, time_samples):
    for dim in np.arange(1, 10):
        H = hamiltonian(dim)
        H.setOptimisableParameters(H.getParameters())
        grads = gen.generateSignalGradient(time_samples)
        hamGrads = H.gradient(time_samples)
        assert hamGrads.shape == (grads.shape[0], grads.shape[1] + 1, dim, dim)
