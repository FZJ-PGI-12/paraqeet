"""Test the unitary fidelity model."""

import numpy as np
import pytest

from cthree.measurement.UnitaryFidelity import UnitaryFidelity
from test.propagation.IdentityPropagation import IdentityPropagation
from test.propagation.RandomPropagation import RandomPropagation


@pytest.fixture
def identityPropagation():
    """Return a mock identity propagation object."""
    return IdentityPropagation()


def test_positivity(randomUnitaryMatrix, randomBasisVectors):
    """Test that the fidelity is always positive."""
    for dim in range(5, 30):
        for i in range(100):
            propagation = RandomPropagation(dim, True)
            gate = randomUnitaryMatrix(dim)
            measurement = UnitaryFidelity(propagation, gate, np.array([1.0]))
            m = measurement.measure()
            assert 0.0 <= m

            # gate must have at least two dimensions
            subDim = np.random.randint(2, dim)
            basisStates = randomBasisVectors(dim, subDim)
            # gate is defined on the subspace.
            gate = randomUnitaryMatrix(subDim)
            measurement = UnitaryFidelity(
                propagation, gate, np.array([1.0]), basisStates
            )
            m = measurement.measure()
            assert 0.0 <= m


def test_positivity_projected(randomUnitaryMatrix):
    """Test the projected positivity of the system."""
    times = np.array([1.0])
    for size in range(5, 30):
        for projectedSize in range(2, size):
            gate = randomUnitaryMatrix(projectedSize)
            propagation = RandomPropagation(size, True)
            measurement = UnitaryFidelity(
                propagation=propagation, gate=gate, times=times
            )

            measurement.restrictSubsystems([size], [projectedSize])
            for _ in range(20):
                m = measurement.measure()
                assert 0.0 <= m <= 1.0


def test_equality(identityPropagation, randomUnitaryMatrix):
    """Test that F(U,U) = 1."""
    for dim in range(2, 30):
        for i in range(100):
            gate = randomUnitaryMatrix(dim)
            np.testing.assert_almost_equal(
                np.conjugate(gate.T) @ gate, np.eye(dim)
            )
            propagation = IdentityPropagation()
            propagation.setInitialState(gate)
            measurement = UnitaryFidelity(propagation, gate, np.array([1.0]))
            m = measurement.measure()
            np.testing.assert_almost_equal(m, 1.0)


def test_projection(identityPropagation, randomBasisVectors):
    """Test the projection of the state vectors."""
    for dim in range(2, 30):
        for i in range(100):
            gate = np.eye(dim)
            init_state = randomBasisVectors(dim + 4, dim)
            propagation = identityPropagation
            propagation.setInitialState(gate)
            measurement = UnitaryFidelity(
                propagation, gate, np.array([1.0]), basis_states=init_state
            )
            m = measurement.measure()
            np.testing.assert_almost_equal(m, 1.0)


def test_incompatible_shape(identityPropagation, randomUnitaryMatrix):
    """Test incompatible shapes for initial and target states.

    A set of initial and target states with different dimensions
    should raise an exception.

    Raises
    ------
    Exception
        Different dimensions for the initial and target states
        raise an exception.

    """
    allDims = np.arange(2, 30)
    for dim in allDims:
        for i in range(100):
            gate = randomUnitaryMatrix(dim)

            # create a propagator of a different dimension
            dimensions = np.delete(allDims, np.where(allDims == dim)[0][0])
            propagation = RandomPropagation(np.random.choice(dimensions), True)

            measurement = UnitaryFidelity(propagation, gate, np.array([1.0]))
            with pytest.raises(Exception):
                measurement.measure()


def test_no_parameters(identityPropagation, randomUnitaryMatrix):
    """Test the no parameter case."""
    state = randomUnitaryMatrix(np.random.randint(2, 30))
    measurement = UnitaryFidelity(identityPropagation, state, np.array([1.0]))
    assert measurement.getParameters() == []
