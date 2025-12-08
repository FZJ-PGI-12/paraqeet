import numpy as np
import pytest

from paraqeet.model.hamiltonian import Hamiltonian
from paraqeet.model.resonator import Resonator
from paraqeet.model.rotating_frame_coupling import RotatingFrameCoupling


@pytest.fixture
def subsystem(random_quantity):
    @pytest.mark.usefixtures("random_quantity")
    def _method(dim: int) -> Hamiltonian:
        frequency = random_quantity(1)
        return Resonator(dim, frequency)

    return _method


@pytest.fixture
def coupling(subsystem, random_quantity):
    @pytest.mark.usefixtures("random_quantity")
    def _method(dim1, dim2) -> RotatingFrameCoupling:
        sub1 = subsystem(dim1)
        sub2 = subsystem(dim2)
        coupling = random_quantity(1, "Hz")
        diff_frequency = random_quantity(1, "Hz")
        return RotatingFrameCoupling(sub1, sub2, coupling, diff_frequency)

    return _method


def test_get_parameters(coupling):
    coup = coupling(np.random.randint(2, 10), np.random.randint(2, 10))
    assert coup.get_parameters() is not None and len(coup.get_parameters()) >= 0


# Tests if get_matrices and get_matrices_one_time return matrices with the correct shape
def test_matrix_dimensions(coupling):
    dim1 = np.random.randint(2, 10)
    dim2 = np.random.randint(2, 10)
    dims = [dim1, dim2]
    coup = coupling(dim1, dim2)

    times = np.linspace(0.0, np.random.randint(1, 10) * np.random.rand(), np.random.randint(1, 10))
    mat = coup.get_RWA_couplings(times[-1])
    for i in range(len(mat)):  # iterate the coupling terms
        assert len(mat[i]) == len(dims)  # two subsystems
        for j in range(len(dims)):
            assert mat[i][j].shape == (dims[j], dims[j])


# Tests if get_gradient and get_gradient_one_time return matrices with the correct shape
def test_gradient_dimensions(coupling):
    dim1 = np.random.randint(2, 10)
    dim2 = np.random.randint(2, 10)
    dims = [dim1, dim2]
    coup = coupling(dim1, dim2)
    coup.set_optimizable_parameters(coup.get_parameters())

    times = np.linspace(0.0, np.random.randint(1, 10) * np.random.rand(), np.random.randint(1, 10))
    mat = coup.get_RWA_gradients_one_time(times[-1])
    for i in range(len(mat[0])):  # iterate the coupling terms
        assert len(mat[0][i]) == len(dims)  # two subsystems
        for j in range(len(dims)):
            assert mat[0][i][j].shape == (dims[j], dims[j])

    coup.set_optimizable_parameters([coup.get_parameters()[1]])
    mat = coup.get_RWA_gradients(times)
    for i in range(len(mat[0])):  # iterate the coupling terms
        assert len(mat[0][i]) == len(dims)  # two subsystems
        for j in range(len(dims)):
            assert mat[0][i][j].shape == (len(times), dims[j], dims[j])
