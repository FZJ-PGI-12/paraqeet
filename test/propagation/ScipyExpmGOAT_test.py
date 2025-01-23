"""Test the Scipy piecewise exponentiation GOAT solver."""

import numpy as np
import pytest

from cthree.propagation.ScipyExpmGOAT import ScipyExpmGOAT
from test.model.DummyModel import DummyModel
from test.model.EmptyHamiltonian import EmptyHamiltonian


@pytest.fixture
def expm():
    """Return a Scipy piecewise exponentiation solver generating function."""

    def _method(dimension, res):
        return ScipyExpmGOAT(DummyModel(EmptyHamiltonian(dimension)), res=res)

    return _method


def test_parameters(expm):
    """Test parameters from the equations of motion."""
    propagation = expm(dimension=np.random.randint(10), res=3)
    assert propagation.get_parameters() == []


def test_resolution(expm):
    """Test the resolution of the solver."""
    for i in range(10):
        propagation = expm(dimension=np.random.randint(2, 100), res=3)
        resolution = np.random.randint(1, 1000)
        propagation.set_resolution(resolution)
        assert propagation.get_resolution() == resolution


def test_state_dimension_vector(randomState, expm, ts):
    """Test dimension and norm of state vectors after propagation.

    The dimension and norm of state vectors should be the same
    after propagation.

    """
    for i in range(10):
        dim = np.random.randint(2, 30)
        state = randomState(dim)
        propagation = expm(dim, res=3)
        propagation.set_initial_state(state)
        propagatedStates = propagation.propagate(ts)
        assert propagatedStates.shape[0] == len(ts)
        assert propagatedStates.shape[1:] == state.shape


def test_state_dimension_matrix(randomMatrix, expm, ts):
    """Test the state matrix after propagation."""
    for i in range(10):
        dim = np.random.randint(2, 30)
        state = randomMatrix(dim, dim)
        propagation = expm(dim, res=3)
        propagation.set_initial_state(state)
        propagatedStates = propagation.propagate(ts)
        assert propagatedStates.shape[0] == len(ts)
        assert propagatedStates.shape[1:] == state.shape
