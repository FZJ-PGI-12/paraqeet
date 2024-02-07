import numpy as np
import pytest

from cthree.measurement.MakhlinFunctional import MakhlinFunctional
from test.propagation.IdentityPropagation import IdentityPropagation
from test.propagation.RandomPropagation import RandomPropagation

iswap = np.array([
    [1.0, 0, 0, 0],
    [0, 0, 1.0j, 0],
    [0, 1.0j, 0, 0],
    [0, 0, 0, 1.0]
])
cnot = np.array([
    [1.0, 0, 0, 0],
    [0, 1.0, 0, 0],
    [0, 0, 0, 1.0],
    [0, 0, 1.0, 0]
])
swap = np.array([
    [1.0, 0, 0, 0],
    [0, 0, 1.0, 0],
    [0, 1.0, 0, 0],
    [0, 0, 0, 1.0]
])
sqrtSwap = np.array([
    [1.0, 0, 0, 0],
    [0, 1.0, 1.0, 0],
    [0, 1.0, 1.00, 0],
    [0, 0, 0, 1.0]
])


# test that the distance is greater or equal 0
def test_positivity():
    propagation = RandomPropagation(4, True)
    times = np.array([1.0])
    measurement = MakhlinFunctional(
        propagation,
        times,
    )

    for i in range(100):
        m = measurement.measure()
        assert 0.0 <= m


# test that local rotations have a distance of 2
def test_local_gates():
    x = np.array([
        [0, 1.0],
        [1.0, 0],
    ])
    y = np.array([
        [0, -1.0j],
        [1.0j, 0],
    ])
    z = np.array([
        [1.0, 0],
        [0, -1.0]
    ])
    gates1 = [np.kron(g, np.eye(2)) for g in [x, y, z]]
    gates2 = [np.kron(np.eye(2), g) for g in [x, y, z]]
    for gate in gates1 + gates2:
        propagation = IdentityPropagation()
        propagation.setInitialState(gate)
        measurement = MakhlinFunctional(propagation, np.array([1.0]))
        m = measurement.measure()
        np.testing.assert_almost_equal(m, 2.0)


# test that perfect entangling gates have a distance of 0
def test_perfect_entanglers():
    for gate in [iswap, cnot]:
        propagation = IdentityPropagation()
        propagation.setInitialState(gate)
        measurement = MakhlinFunctional(propagation, np.array([1.0]))
        m = measurement.measure()
        np.testing.assert_almost_equal(m, 0.0)


# test that some special gates generate the expected Makhlin invariants
def test_invariants():
    expectedInvariants = [
        [np.eye(4), [1, 0, 3]],
        [iswap, [0, 0, -1]],
        [cnot, [0, 0, 1]],
        [swap, [-1, 0, -3]],
    ]

    propagation = IdentityPropagation()
    for (gate, invariants) in expectedInvariants:
        propagation.setInitialState(gate)
        measurement = MakhlinFunctional(propagation, np.array([1.0]), np.array(invariants))
        m = measurement.measure()
        np.testing.assert_almost_equal(m, 0.0)


# Test that all propagators which are not 4-dimensional raise an exception
def test_incompatible_shape():
    incompatibleDimensions = np.delete(np.arange(2, 100), 2)
    for dim in incompatibleDimensions:
        propagation = RandomPropagation(dim, True)
        measurement = MakhlinFunctional(propagation, np.array([1.0]))
        with pytest.raises(Exception):
            m = measurement.measure()
