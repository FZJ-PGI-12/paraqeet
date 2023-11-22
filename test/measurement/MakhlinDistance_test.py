import numpy as np

from cthree.measurement.MakhlinDistance import MakhlinDistance
from cthree.measurement.StateTransferFidelity import StateTransferFidelity
from test.propagation.IdentityPropagation import IdentityPropagation
from test.propagation.RandomPropagation import RandomPropagation


# test that the distance is greater or equal 0
def test_positivity():
    for size in range(2, 30):
        propagation = RandomPropagation(size, False)
        times = np.array([1.0])
        measurement = MakhlinDistance(
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
        measurement = MakhlinDistance(propagation, np.array([1.0]))
        m = measurement.measure()
        np.testing.assert_almost_equal(m, 2.0)


# test that perfect entangling gates have a distance of 0
def test_perfect_entanglers():
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
    for gate in [iswap, cnot]:
        propagation = IdentityPropagation()
        propagation.setInitialState(gate)
        measurement = MakhlinDistance(propagation, np.array([1.0]))
        m = measurement.measure()
        np.testing.assert_almost_equal(m, 0.0)
