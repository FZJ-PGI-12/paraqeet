import pytest

import numpy as np

from typing import List

from cthree.model.Hamiltonian import Hamiltonian
from cthree.model.Model import Model
from cthree.Quantity import Quantity
from cthree.signal.Generator import Generator

LEN_SIG = 20
DIMS = 10


@pytest.fixture
def ts():
    return np.linspace(0, 1e-9, LEN_SIG)


@pytest.fixture
def identity():
    return np.identity(DIMS)


@pytest.fixture
def model():
    return dummy_model(DummyHamiltonian())


class DummyHamiltonian(Hamiltonian):
    def __init__(self):
        super().__init__([], None, None, Generator())

    def getMatrix(self, t: np.ndarray) -> np.ndarray:
        return np.zeros((LEN_SIG, DIMS, DIMS))


class dummy_model(Model):
    def __init__(self, hamiltonian: Hamiltonian):
        super().__init__(hamiltonian)

    def getParameters(self) -> List[Quantity]:
        pass

    def getEquationOfMotion(self, t: np.ndarray) -> np.ndarray:
        return -1.0j * self._hamiltonian.getMatrix(t) * (t[1] - t[0])
