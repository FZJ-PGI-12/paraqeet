import numpy as np
import pytest

from paraqeet.model.master_equation import MasterEquation
from paraqeet.model.resonator import Resonator
from paraqeet.quantity import Quantity


@pytest.fixture
def hamiltonian():
    def _method(num_fock):
        FREQ = 4.8e9 * 2 * np.pi
        return Resonator(
            frequency=Quantity(FREQ, 0.8 * FREQ, 1.2 * FREQ),
            num_fock=num_fock,
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
        jump_ops = hamil.get_jump_operators()
        return MasterEquation(hamil.get_hamiltonian, hamil.get_hamiltonian_and_gradient, jump_ops)

    return _method


def test_create_dense_matrix(open_system):
    for dim in range(3, 6):
        random_time_vector = np.linspace(0.0, np.random.randint(1, 10) * np.random.rand(), np.random.randint(1, 10))

        system = open_system(dim)
        system.sparse_superop = True
        assert system.sparse_superop is True
        matrix = system.get_value(random_time_vector)
        assert matrix.shape == (len(random_time_vector), dim**2, dim**2)
