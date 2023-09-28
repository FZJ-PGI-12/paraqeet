import pytest

import numpy as np

from typing import List

from cthree.model.Hamiltonian import Hamiltonian
from cthree.model.Model import Model
from cthree.Quantity import Quantity
from cthree.signal.Generator import Generator
from test.model.DummyModel import DummyModel
from test.model.EmptyHamiltonian import EmptyHamiltonian

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
    return DummyModel(EmptyHamiltonian(DIMS))
