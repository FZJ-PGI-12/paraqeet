"""Test the Euler propagation model."""

import pytest
import numpy as np

from cthree.propagation.Euler import Euler
from test.model.DummyModel import DummyModel
from test.model.EmptyHamiltonian import EmptyHamiltonian


@pytest.fixture
def euler():
    """Return a Euler propagation model generating method."""

    def _method(dimension):
        return Euler(DummyModel(EmptyHamiltonian(dimension)))

    return _method


def test_parameters(euler):
    """Test parameters of the model."""
    assert euler(2).getParameters() == []


def test_state_dimension_vector(randomState, euler, ts):
    """Test the dimension and the norm of the state vector.

    The dimension and the norm should be the same.

    """
    for i in range(10):
        dim = np.random.randint(2, 30)
        state = randomState(dim)
        propagation = euler(dim)
        propagation.setInitialState(state)
        propagatedStates = propagation.propagate(ts)
        assert propagatedStates.shape[0] == len(ts)
        assert propagatedStates.shape[1:] == state.shape


def test_state_dimension_matrix(randomMatrix, euler, ts):
    """Test the state dimension matrix."""
    for i in range(10):
        dim = np.random.randint(2, 30)
        state = randomMatrix(dim, dim)
        propagation = euler(dim)
        propagation.setInitialState(state)
        propagatedStates = propagation.propagate(ts)
        assert propagatedStates.shape[0] == len(ts)
        assert propagatedStates.shape[1:] == state.shape
