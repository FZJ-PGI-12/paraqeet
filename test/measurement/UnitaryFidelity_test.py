import numpy as np
import pytest

from cthree.measurement.UnitaryFidelity import UnitaryFidelity
from test.propagation.IdentityPropagation import IdentityPropagation
from test.propagation.RandomPropagation import RandomPropagation


@pytest.fixture
def identityPropagation():
    return IdentityPropagation()


# test that the fidelity is always positive
def test_positivity(randomUnitaryMatrix, randomBasisVectors):
    for dim in range(5, 30):
        for i in range(100):
            propagation = RandomPropagation(dim, True)
            gate = randomUnitaryMatrix(dim)
            measurement = UnitaryFidelity(propagation, gate, np.array([1.0]))
            m = measurement.measure()
            assert 0.0 <= m

            basisStates = randomBasisVectors(dim, np.random.randint(1, dim))
            measurement = UnitaryFidelity(propagation, gate, np.array([1.0]), basisStates)
            m = measurement.measure()
            assert 0.0 <= m


# test that F(U,U) = 1
def test_equality(identityPropagation, randomUnitaryMatrix):
    for dim in range(2, 30):
        for i in range(100):
            gate = randomUnitaryMatrix(dim)
            np.testing.assert_almost_equal(np.conjugate(gate.T) @ gate, np.eye(dim))
            propagation = IdentityPropagation()
            propagation.setInitialState(gate)
            measurement = UnitaryFidelity(propagation, gate, np.array([1.0]))
            m = measurement.measure()
            np.testing.assert_almost_equal(m, 1.0)


# Test that a propagator that has a different dimension than the ideal gate raises an exception
def test_incompatible_shape(identityPropagation, randomUnitaryMatrix):
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


def test_no_parameters(identityPropagation, randomState):
    state = randomState(np.random.randint(2, 30))
    measurement = UnitaryFidelity(identityPropagation, state, state, np.array([1.0]))
    assert measurement.getParameters() == []
