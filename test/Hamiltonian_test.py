import pytest
import numpy as np

from cthree.model.Hamiltonian import Hamiltonian
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
    sigmaZ = np.array([[1.0, 0], [0, -1]])
    sigmaX = np.array([[0.0, 1], [1, 0]])
    drift = FREQ / 2 * sigmaZ
    return Hamiltonian(subsystems=[drift], drives=[sigmaX], generator=gen)

def test_getMatrix(ham, time_samples):
    hams = ham.getMatrix(time_samples)
    assert hams.shape == time_samples.shape + (2, 2)

def test_gradient(gen, ham, time_samples):
    grads = gen.generateSignalGradient(time_samples)
    hamGrads = ham.gradient(time_samples)
    assert hamGrads.shape == grads.shape + (2, 2)