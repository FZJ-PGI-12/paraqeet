import pytest

import numpy as np
from scipy.stats import unitary_group

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
def randomUnitaryMatrix():
    """
    Returns a method that generates random unitary matrices for a given dimension.
    """

    def _method(dim):
        return unitary_group.rvs(dim)

    return _method
