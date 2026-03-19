"""Test the Scipy piecewise exponentiation solver."""

import numpy as np
import pytest

from paraqeet.propagation.scipy_expm import ScipyExpm
from paraqeet.propagation.utils import convert_dm_to_vec, convert_vec_to_dm
from tests.model.dummy_model import DummyEquationsOfMotion
from tests.model.empty_hamiltonian import EmptySystem
from tests.propagation.test_common_propagation import check_propagation


@pytest.fixture
def expm():
    """Return a Scipy piecewise exponentitation solver generating method."""

    def _method(dimension, resolution):
        return ScipyExpm(
            eom_func=DummyEquationsOfMotion(EmptySystem(dimension)).get_value,
            resolution=resolution,
            initial_state=np.eye(dimension, dtype=np.complex128),
        )

    return _method


def test_resolution(expm):
    """Test the resolution after propagation."""
    for i in range(10):
        propagation = expm(dimension=np.random.randint(2, 100), resolution=3)
        resolution = np.random.randint(1, 1000)
        propagation.resolution = resolution
        assert propagation.resolution == resolution


def test_state_dimension_vector(random_state, expm, ts):
    """Test the dimension and the norm of state vectors.

    The dimension and norm of state vectors should be the
    same after propagation.

    """
    for _ in range(10):
        dim = np.random.randint(2, 30)
        state = random_state(dim)
        propagation = expm(dim, resolution=3)
        propagation.initial_state = state
        propagated_states = propagation.propagate(ts)
        assert propagated_states.shape[0] == len(ts)
        assert propagated_states.shape[1:] == state.shape


def test_state_dimension_rect_matrix(random_matrix, expm, ts):
    """Test the state matrix after propagation."""
    for _ in range(10):
        basis = np.random.randint(2, 10)
        dim = basis + np.random.randint(1, 3)
        state = random_matrix(dim, basis)  # rect matrix with dim>basis
        propagation = expm(dim, resolution=3)
        propagation.initial_state = state
        propagated_states = propagation.propagate(ts)
        assert propagated_states.shape[0] == len(ts)
        assert propagated_states.shape[1:] == state.shape


def test_state_dimension_square_matrix(expm, ts):
    """Test for a batch of initial states for propagation."""
    for _ in range(5):
        basis = np.random.randint(2, 10)
        dim = basis + np.random.randint(1, 3)
        state = np.eye(dim, dtype=np.complex128)
        propagation = expm(dim, resolution=3)
        propagation.initial_state = state
        propagated_states = propagation.propagate(ts)
        assert propagated_states.shape[0] == len(ts)
        assert propagated_states.shape[1:] == state.shape


def test_state_dimension_matrix_open(random_matrix, expm, ts):
    """Test the state matrix after propagation."""
    for _ in range(10):
        dim = np.random.randint(2, 10)
        state = random_matrix(dim, dim)
        propagation = expm(dim**2, resolution=3)
        propagation.initial_state = convert_dm_to_vec(state, dim)
        propagated_states = propagation.propagate(ts)
        assert propagated_states.shape[0] == len(ts)
        assert propagated_states.shape[1:] == convert_dm_to_vec(state, dim).shape
        assert convert_vec_to_dm(propagated_states[-1], dim).shape == state.shape


def test_initial_state(eom):
    """Test whether the initial state is set.

    Raises
    ------
    ConfigurationException
        If the initial state is not set.

    """
    dim = np.random.randint(2, 10)
    m = eom(dim)
    propagation = ScipyExpm(eom_func=m, resolution=3, initial_state=np.eye(dim, dtype=np.complex128))

    check_propagation(propagation, dim)


def test_construct_times(eom):
    """Test the construction times for the model."""
    res = 100e9

    propagation = ScipyExpm(eom_func=eom, resolution=res, initial_state=np.eye(10, dtype=np.complex128))

    # Test if times array is constructed correctly for a 1ns list
    t_start = 1e-9
    t_final = 2e-9

    time = np.array([t_start, t_final])
    steps = int(np.ceil((t_final - t_start) * res))
    full_times = np.linspace(t_start, t_final, steps, endpoint=False)
    times, dt = propagation._construct_times(time, 1)

    assert np.allclose(times, full_times)
    assert np.isclose(dt, 1 / res)

    # Test if times array is constructed correctly if steps < 2
    t_start = 1e-9
    t_final = t_start + (1 / res)

    time = np.array([t_start, t_final])
    steps = int(np.ceil((t_final - t_start) * res))
    full_times = np.linspace(t_start, t_final, steps, endpoint=False)
    times, dt = propagation._construct_times(time, 1)

    assert np.allclose(time, full_times)
    assert np.isclose(dt, 1 / res)
