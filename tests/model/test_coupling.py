"""Test the coupling model."""

import numpy as np
import pytest

from paraqeet.model.coupling import TwoBodyCoupling
from paraqeet.model.transmon import Transmon
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

    def _method(dim1: int, dim2: int, is_longitudinal: bool, use_rwa: bool = False):
        transmonA = transmon.get(dim1)
        transmonB = transmon.get(dim2)
        coupling = TwoBodyCoupling(
            subsystem_A=transmonA,
            subsystem_B=transmonB,
            is_longitudinal=is_longitudinal,
            use_rwa=use_rwa,
            coefficient=Quantity(COUPLINGSTR, 0.8 * COUPLINGSTR, 1.2 * COUPLINGSTR, "Hz"),
        )

        return coupling

    return _method


def test_get_matrices(coupling):
    """Test the get matrice method workings."""
    for _ in range(10):
        dim1 = np.random.randint(2, 7)
        dim2 = np.random.randint(2, 7)
        dims = [dim1, dim2]

        # Test shape for Longitudinal coupling
        coup = coupling(dim1, dim2, is_longitudinal=True)
        coup_hams = coup.get_couplings()
        for term in coup_hams:
            for i, ops in enumerate(term):
                assert np.shape(ops) == (dims[i], dims[i])

        # Test for RWA
        coup = coupling(dim1, dim2, is_longitudinal=False, use_rwa=True)
        coup_hams = coup.get_couplings()
        for term in coup_hams:
            for i, ops in enumerate(term):
                assert np.shape(ops) == (dims[i], dims[i])

        # Test shape for Transverse coupling
        coup = coupling(dim1, dim2, is_longitudinal=False)
        coup_hams = coup.get_couplings()
        for term in coup_hams:
            for i, ops in enumerate(term):
                assert np.shape(ops) == (dims[i], dims[i])


def test_gradient_shape(coupling):
    """Test the shape of the gradient."""
    # TODO: Test if the coupling is not optimized
    dim1 = np.random.randint(2, 7)
    dim2 = np.random.randint(2, 7)
    dims = [dim1, dim2]
    coup = coupling(dim1, dim2, is_longitudinal=False)
    grads = coup.get_coupling_gradients()
    for grad in grads:
        for term in grad:
            for i, ops in enumerate(term):
                assert np.size(ops) == 0
    coup.set_optimizable_parameters(coup.get_parameters())
    grads = coup.get_coupling_gradients()
    for grad in grads:
        for term in grad:
            for i, ops in enumerate(term):
                assert np.shape(ops) == (dims[i], dims[i])

    # Test with RWA
    dim1 = np.random.randint(2, 7)
    dim2 = np.random.randint(2, 7)
    dims = [dim1, dim2]
    coup = coupling(dim1, dim2, is_longitudinal=False, use_rwa=True)
    coup.set_optimizable_parameters(coup.get_parameters())
    grads = coup.get_coupling_gradients()
    for grad in grads:
        for term in grad:
            for i, ops in enumerate(term):
                assert np.shape(ops) == (dims[i], dims[i])
