"""Test the coupling model."""

import numpy as np
import pytest

from paraqeet.hamiltonian.coupling import Coupling
from paraqeet.hamiltonian.transmon import TransmonHamiltonian
from paraqeet.quantity import Quantity

COUPLINGSTR = 25e6 * 2 * np.pi


@pytest.fixture
def transmon_parameters():
    """Return a random parameter object for transmons."""

    class RandomParameters:
        def get(self):
            freq = np.random.uniform(5.5, 6.0) * 1e9 * 2 * np.pi
            anharm = -np.random.uniform(2.0, 2.4) * 1e7 * 2 * np.pi
            return (freq, anharm)

    return RandomParameters()


@pytest.fixture
def transmon(transmon_parameters):
    """Return a transmon created from the given parameters."""

    class CreateTransmon:
        def get(self, num_levels):
            freq, anharm = transmon_parameters.get()
            transmon_hamiltonian = TransmonHamiltonian(
                num_levels=num_levels,
                frequency=Quantity(freq, 0.8 * freq, 1.2 * freq),
                anharmonicity=Quantity(anharm, 1.2 * anharm, 0.8 * anharm),
            )
            return transmon_hamiltonian

    return CreateTransmon()


@pytest.fixture
def coupling(transmon):
    """Return a coupling generator method."""

    def _method(dim1: int, dim2: int, add_hermitian: bool):
        transmon_a = transmon.get(dim1)
        transmon_b = transmon.get(dim2)
        if add_hermitian:
            coupling_op = np.kron(
                transmon_a.annihilation_op,
                transmon_b.annihilation_op.conj().T,
            )
        else:
            coupling_op = np.kron(
                transmon_a.annihilation_op + transmon_a.annihilation_op.conj().T,
                transmon_b.annihilation_op + transmon_b.annihilation_op.conj().T,
            )

        coupling = Coupling(
            coupling_op=coupling_op,
            g_abs=Quantity(COUPLINGSTR, 0.8 * COUPLINGSTR, 1.2 * COUPLINGSTR, "Hz"),
            add_hermitian=add_hermitian,
        )

        return coupling

    return _method


def test_get_matrices(coupling):
    """Test the get matrice method workings."""
    for _ in range(10):
        dim1 = np.random.randint(2, 7)
        dim2 = np.random.randint(2, 7)
        total_dim = dim1 * dim2

        # Test add Hermitian
        coup = coupling(dim1, dim2, add_hermitian=False)
        coup_ham = coup.get_value(np.array([0.0]))
        assert np.shape(coup_ham) == (1, total_dim, total_dim)
        # Test shape with add Hermitian
        coup = coupling(dim1, dim2, add_hermitian=True)
        coup_ham = coup.get_value(np.array([0.0]))
        assert np.shape(coup_ham) == (1, total_dim, total_dim)


def test_gradient_shape(coupling):
    """Test the shape of the gradient."""
    # TODO: Test if the coupling is not optimized
    times = np.array([0.0, 1e-9, 2e-9, 3e-9])
    dim1 = np.random.randint(2, 7)
    dim2 = np.random.randint(2, 7)
    total_dim = dim1 * dim2
    coup = coupling(dim1, dim2, add_hermitian=False)
    grads = coup.get_gradient(times)
    assert np.shape(grads) == (*times.shape, 0, total_dim, total_dim)
    coup.set_optimizable_parameters(coup.get_parameters())
    grads = coup.get_gradient(times)
    assert np.shape(grads) == (*times.shape, 2, total_dim, total_dim)
    # Test with RWA
    dim1 = np.random.randint(2, 7)
    dim2 = np.random.randint(2, 7)
    total_dim = dim1 * dim2
    coup = coupling(dim1, dim2, add_hermitian=True)
    coup.set_optimizable_parameters(coup.get_parameters())
    grads = coup.get_gradient(times)
    assert np.shape(grads) == (*times.shape, 2, total_dim, total_dim)
