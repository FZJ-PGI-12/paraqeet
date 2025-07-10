"""Test the Scipy piecewise exponentiation GOAT solver."""

import numpy as np
import pytest

from paraqeet.propagation.scipy_expm_goat import ScipyExpmGOAT
from test.model.dummy_model import DummyEquationsOfMotion
from test.model.empty_hamiltonian import EmptyHamiltonian
from test.propagation.common_propagation_tests import needs_initial_state


@pytest.fixture
def expm():
    """Return a Scipy piecewise exponentiation solver generating function."""

    def _method(dimension, res):
        return ScipyExpmGOAT(DummyEquationsOfMotion(EmptyHamiltonian(dimension)), res=res)

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
        propagation.resolution = resolution
        assert propagation.resolution == resolution


def test_state_dimension_vector(random_state, expm, ts):
    """Test dimension and norm of state vectors after propagation.

    The dimension and norm of state vectors should be the same
    after propagation.

    """
    for i in range(10):
        dim = np.random.randint(2, 30)
        state = random_state(dim)
        propagation = expm(dim, res=3)
        propagation.set_initial_state(state)
        propagatedStates = propagation.propagate(ts)
        assert propagatedStates.shape[0] == len(ts)
        assert propagatedStates.shape[1:] == state.shape


def test_state_dimension_matrix(random_matrix, expm, ts):
    """Test the state matrix after propagation."""
    for i in range(10):
        dim = np.random.randint(2, 30)
        state = random_matrix(dim, dim)
        propagation = expm(dim, res=3)
        propagation.set_initial_state(state)
        propagatedStates = propagation.propagate(ts)
        assert propagatedStates.shape[0] == len(ts)
        assert propagatedStates.shape[1:] == state.shape


def test_needs_initial_state(random_state, expm):
    for dim in range(2, 10):
        propagation = expm(dim, res=3)
        needs_initial_state(propagation, dim)
