import pytest

import numpy as np

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


@pytest.fixture
def randomState():
    """
    Returns a method that generates random normalised states for a given dimension.
    """

    def _method(dimension):
        state = np.random.random(dimension) + 1j * np.random.random(dimension)
        return state / np.sqrt(np.vdot(state, state))

    return _method
