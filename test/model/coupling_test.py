"""Test the coupling model."""

import pytest
import numpy as np

from cthree.quantity import Quantity
from cthree.model.coupling import Coupling
from cthree.model.transmon import Transmon

COUPLINGSTR = 25e6 * 2 * np.pi
LEN_SIG = 101


@pytest.fixture
def time_samples():
    """Generate time samples from the given signal length.

    The given signal length `LEN_SIG` is a module level global variable.

    """
    return np.linspace(0, 10e-9, LEN_SIG)


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
        def get(self, dimension):
            freq, anharm = transmon_parameters.get()
            transmon = Transmon(
                dimension=dimension,
                frequency=Quantity(freq, 0.8 * freq, 1.2 * freq),
                anharmonicity=Quantity(anharm, 1.2 * anharm, 0.8 * anharm),
            )
            return transmon

    return CreateTransmon()


@pytest.fixture
def coupling(transmon):
    """Return a coupling generator method."""

    def _method(dim1: int, dim2: int, isLongitudinal: bool, useRWA: bool = False):
        transmon1 = transmon.get(dim1)
        transmon2 = transmon.get(dim2)
        coupling = Coupling(
            [transmon1, transmon2],
            is_longitudinal=isLongitudinal,
            useRWA=useRWA,
            coefficient=Quantity(COUPLINGSTR, 0.8 * COUPLINGSTR, 1.2 * COUPLINGSTR, "Hz"),
        )

        return coupling

    return _method


def test_get_matrices_one_time(coupling):
    """Test shape of Matrix produced by the coupling Hamiltonian."""
    for _ in range(10):
        dim1 = np.random.randint(2, 7)
        dim2 = np.random.randint(2, 7)
        dims = [dim1, dim2]

        # Test shape for Longitudinal coupling
        coup = coupling(dim1, dim2, isLongitudinal=True)
        coup_hams = coup.get_matrices_one_time(0)
        for term in coup_hams:
            for i, ops in enumerate(term):
                assert np.shape(ops) == (dims[i], dims[i])

        # Test Longitudinal couplings are diagonal
        for term in coup_hams:
            for i, ops in enumerate(term):
                numNonZero = np.count_nonzero(ops - np.diag(np.diagonal(ops)))
                assert numNonZero == 0

        # Test for RWA
        coup = coupling(dim1, dim2, isLongitudinal=False, useRWA=True)
        coup_hams = coup.get_matrices_one_time(0)
        for term in coup_hams:
            for i, ops in enumerate(term):
                assert np.shape(ops) == (dims[i], dims[i])

        # Test shape for Transverse coupling
        coup = coupling(dim1, dim2, isLongitudinal=False)
        coup_hams = coup.get_matrices_one_time(0)
        for term in coup_hams:
            for i, ops in enumerate(term):
                assert np.shape(ops) == (dims[i], dims[i])


def test_get_matrices(coupling, time_samples):
    """Test the get matrice method workings."""
    for _ in range(10):
        dim1 = np.random.randint(2, 7)
        dim2 = np.random.randint(2, 7)
        dims = [dim1, dim2]

        # Test shape for Longitudinal coupling
        coup = coupling(dim1, dim2, isLongitudinal=True)
        coup_hams = coup.get_matrices(time_samples)
        for term in coup_hams:
            for i, ops in enumerate(term):
                assert np.shape(ops) == time_samples.shape + (dims[i], dims[i])

        # Test for RWA
        coup = coupling(dim1, dim2, isLongitudinal=False, useRWA=True)
        coup_hams = coup.get_matrices(time_samples)
        for term in coup_hams:
            for i, ops in enumerate(term):
                assert np.shape(ops) == time_samples.shape + (dims[i], dims[i])

        # Test shape for Transverse coupling
        coup = coupling(dim1, dim2, isLongitudinal=False)
        coup_hams = coup.get_matrices(time_samples)
        for term in coup_hams:
            for i, ops in enumerate(term):
                assert np.shape(ops) == time_samples.shape + (dims[i], dims[i])


def test_gradient_shape(coupling, time_samples):
    """Test the shape of the gradient."""
    # Test if the coupling is not optimized
    dim1 = np.random.randint(2, 7)
    dim2 = np.random.randint(2, 7)
    dims = [dim1, dim2]
    coup = coupling(dim1, dim2, isLongitudinal=False)
    grads = coup.gradient(time_samples)
    for grad in grads:
        for term in grad:
            for i, ops in enumerate(term):
                assert np.size(ops) == 0
    coup.set_optimisable_parameters(coup.get_parameters())
    grads = coup.gradient(time_samples)
    for grad in grads:
        for term in grad:
            for i, ops in enumerate(term):
                assert np.shape(ops) == time_samples.shape + (dims[i], dims[i])

    # Test with RWA
    dim1 = np.random.randint(2, 7)
    dim2 = np.random.randint(2, 7)
    dims = [dim1, dim2]
    coup = coupling(dim1, dim2, isLongitudinal=False, useRWA=True)
    coup.set_optimisable_parameters(coup.get_parameters())
    grads = coup.gradient(time_samples)
    for grad in grads:
        for term in grad:
            for i, ops in enumerate(term):
                assert np.shape(ops) == time_samples.shape + (dims[i], dims[i])
