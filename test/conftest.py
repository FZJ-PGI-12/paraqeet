"""Testing the configuration functions."""

import pytest

import numpy as np
from scipy.stats import unitary_group

from paraqeet.quantity import Quantity
from test.model.dummy_model import DummyEquationsOfMotion
from test.model.empty_hamiltonian import EmptyHamiltonian

LEN_SIG = 20


@pytest.fixture
def ts():
    """Return evenly spaced numbers according to the given signal length.

    The given signal length `LEN_SIG` is a module level global variable.

    """
    return np.linspace(0, 1e-9, LEN_SIG)


@pytest.fixture
def identity():
    """Return an identity array generation function."""

    def _method(dimension):
        return np.identity(dimension)

    return _method


@pytest.fixture
def model():
    """Return a dummy model generation function."""

    def _method(dimension):
        return DummyEquationsOfMotion(EmptyHamiltonian(dimension))

    return _method


@pytest.fixture
def random_state():
    """Return a random state generating method.

    Generates random normalised states for a given dimension.

    """

    def _method(dimension):
        state = np.random.random(dimension) + 1j * np.random.random(dimension)
        return state / np.sqrt(np.vdot(state, state))

    return _method


@pytest.fixture
def random_matrix():
    """Return a random matrix generating method.

    Generates random matrix for given dimensions n and m.
    The matrix is normalised to have trace 1.

    """

    def _method(n, m):
        state = np.random.random(size=(n, m))
        +1j * np.random.random(size=(n, m))
        return state / np.trace(state)

    return _method


@pytest.fixture
def random_unitary_matrix():
    """Return a random unitary matrix generating method.

    Generates random unitary matrices for a given dimension.

    """

    def _method(dim):
        return unitary_group.rvs(dim)

    return _method


@pytest.fixture
def random_basis_vectors():
    """Return a random basis vector generating method.

    Generates N vectors, each with 0 everywhere except a 1 at a random index.
    All vectors will be orthogonal.

    """

    def _method(dim, N):
        v = np.zeros((dim, N))
        indices = np.random.choice(np.arange(0, dim), N, replace=False)
        for i in range(N):
            v[indices[i], i] = 1
        return v

    return _method


@pytest.fixture
# helper functions
def random_quantity(random_quantity_for_values):
    """Generate a quantity with N positive and negative numbers.

    Each quantity is with the same order of magnitude
    which is chosen randomly between 1e-10 and 1e10.

    """

    def _method(N: int):
        magnitude = np.power(10.0, np.random.randint(-10, 10))
        values = (2 * np.random.random(N) - 1) * magnitude
        return random_quantity_for_values(values)

    return _method


@pytest.fixture
def random_quantity_for_values(random_limits_for_quantity):
    """Generate a quantity from the given array of values.

    Generates while making sure that the limits are set correctly.

    """

    def _method(values: np.ndarray):
        limits = random_limits_for_quantity(values)
        return Quantity(values, min_value=limits[0], max_value=limits[1], unit="")

    return _method


@pytest.fixture
def random_limits_for_quantity():
    """Return random limits for quantities.

    Returns random but valid minimum and maximum values
    for the given value array while taking acount for negative values.

    """

    def _method(values: np.ndarray):
        if len(values.shape) == 0:
            # scalar quantity
            if values == 0.0:
                min_value = np.array(-1.0)
                max_value = np.array(+1.0)
            elif values < 0:
                min_value = (np.random.random() + 1.1) * values
                max_value = np.random.random() * values
            else:
                min_value = np.random.random() * values
                max_value = (np.random.random() + 1.1) * values
            return min_value, max_value
        else:
            # list quantity
            min_values, max_values = (
                np.zeros_like(values),
                np.zeros_like(values),
            )
            for i, v in enumerate(values):
                if v == 0.0:
                    min_values[i] = -1
                    max_values[i] = +1
                elif v < 0:
                    min_values[i] = (np.random.random() + 1.1) * v
                    max_values[i] = np.random.random() * v
                else:
                    min_values[i] = np.random.random() * v
                    max_values[i] = (np.random.random() + 1.1) * v
            return min_values, max_values

    return _method


@pytest.fixture
def random_from_list():
    """Returns one or more random values from the given list, excluding specific values from the list."""

    def _method(selectFrom: np.ndarray, excluded: np.ndarray | float, numSelected: int = 1):
        if np.isscalar(excluded):
            excluded = np.array([excluded])
        selectable = np.delete(selectFrom, excluded)
        return np.random.choice(selectable)

    return _method
