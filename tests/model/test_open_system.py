import numpy as np
import pytest

from paraqeet.model.open_system import OpenSystem
from paraqeet.model.resonator import Resonator
from paraqeet.quantity import Quantity


@pytest.fixture
def hamiltonian():
    def _method(dimension):
        FREQ = 4.8e9 * 2 * np.pi
        return Resonator(
            frequency=Quantity(FREQ, 0.8 * FREQ, 1.2 * FREQ),
            dimension=dimension,
        )

    return _method


@pytest.fixture
def open_system(
    hamiltonian,
):
    def _method(dimension):
        hamil = hamiltonian(dimension)
        hamil.t1 = Quantity(1e-9, 1e-9, 100e-6)
        hamil.temp = Quantity(10e-3, 1e-3, 50e-3)
        hamil.t2star = Quantity(10e-9, 1e-9, 100e-6)
        return OpenSystem(hamil)

    return _method


def test_create_dense_matrix(open_system):
    for dim in range(3, 6):
        random_time_vector = np.linspace(0.0, np.random.randint(1, 10) * np.random.rand(), np.random.randint(1, 10))

        system = open_system(dim)
        system.sparse_superop = True
        assert system.sparse_superop is True
        matrix = system.get_value(random_time_vector)
        assert matrix.shape == (len(random_time_vector), dim**2, dim**2)


def test_create_sparse_matrix(open_system):
    for dim in range(3, 6):
        random_time_vector = np.linspace(0.0, np.random.randint(1, 10) * np.random.rand(), np.random.randint(1, 10))

        system = open_system(dim)
        system.sparse_superop = False
        assert system.sparse_superop is False
        matrix = system.get_value(random_time_vector)
        assert matrix.shape == (len(random_time_vector), dim**2, dim**2)


def test_has_parameters(open_system):
    system = open_system(np.random.randint(2, 10))
    assert system.get_parameters() is not None
    assert len(system.get_parameters()) >= 0
