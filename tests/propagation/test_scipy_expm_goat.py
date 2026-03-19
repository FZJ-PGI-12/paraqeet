"""Test the Scipy piecewise exponentiation GOAT solver."""

import numpy as np
import pytest

from paraqeet.propagation.scipy_expm_goat import ScipyExpmGOAT
from tests.model.dummy_model import DummyEquationsOfMotion
from tests.model.empty_hamiltonian import EmptySystem
from tests.propagation.test_common_propagation import check_propagation


@pytest.fixture
def expm():
    """Return a Scipy piecewise exponentiation solver generating function."""

    def _method(dimension, resolution):
        return ScipyExpmGOAT(DummyEquationsOfMotion(EmptySystem(dimension)), resolution=resolution)

    return _method


def test_parameters(expm):
    """Test parameters from the equations of motion."""
    propagation = expm(dimension=np.random.randint(10), resolution=3)
    assert propagation.get_parameters() == []


def test_resolution(expm):
    """Test the resolution of the solver."""
    for _ in range(10):
        propagation = expm(dimension=np.random.randint(2, 100), resolution=3)
        resolution = np.random.randint(1, 1000)
        propagation.resolution = resolution
        assert propagation.resolution == resolution


def test_state_dimension_vector(random_state, expm, ts):
    """Test dimension and norm of state vectors after propagation.

    The dimension and norm of state vectors should be the same
    after propagation.

    """
    for _ in range(10):
        dim = np.random.randint(2, 10)
        state = random_state(dim)
        propagation = expm(dim, resolution=3)
        propagation.set_initial_state(state)
        propagated_states = propagation.propagate(ts)
        assert propagated_states.shape[0] == len(ts)
        assert propagated_states.shape[1:] == state.shape + (1,)


@pytest.mark.parametrize("is_open", [True, False])
def test_state_dimension_matrix(random_matrix, expm, ts, is_open):
    """Test the state matrix after propagation."""
    for _ in range(10):
        dim = np.random.randint(2, 10)
        state = random_matrix(dim, dim)
        if is_open:
            propagation = expm(dim**2, resolution=3)
            propagation.is_open = True
        else:
            propagation = expm(dim, resolution=3)
        propagation.set_initial_state(state)
        propagated_states = propagation.propagate(ts)
        assert propagated_states.shape[0] == len(ts)
        assert propagated_states.shape[1:] == state.shape


def test_needs_initial_state(random_state, expm):
    for dim in range(2, 10):
        propagation = expm(dim, resolution=3)
        check_propagation(propagation, dim)
