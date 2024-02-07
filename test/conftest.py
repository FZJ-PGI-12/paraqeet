import pytest

import numpy as np

from test.model.DummyModel import DummyModel
from test.model.EmptyHamiltonian import EmptyHamiltonian

LEN_SIG = 20


@pytest.fixture
def ts():
    return np.linspace(0, 1e-9, LEN_SIG)


@pytest.fixture
def identity():
    def _method(dimension):
        return np.identity(dimension)

    return _method


@pytest.fixture
def model():
    def _method(dimension):
        return DummyModel(EmptyHamiltonian(dimension))

    return _method


@pytest.fixture
def randomState():
    """
    Returns a method that generates random normalised states for a given dimension.
    """

    def _method(dimension):
        state = np.random.random(dimension) + 1j * np.random.random(dimension)
        return state / np.sqrt(np.vdot(state, state))

    return _method

@pytest.fixture
def randomMatrix():
    """
    Returns a method that generates random matrix for given dimensions n and m. The matrix is normalised to have
    trace 1.
    """

    def _method(n, m):
        state = np.random.random(size=(n, m)) + 1j * np.random.random(size=(n, m))
        return state / np.trace(state)

    return _method
