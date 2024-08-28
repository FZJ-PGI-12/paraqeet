import numpy as np

from cthree.measurement.UnitaryFidelity import UnitaryFidelity
from cthree.measurement.StateTransferFidelity import StateTransferFidelity
from test.propagation.RandomPropagation import RandomPropagation


def randomState(dimension):
    state = np.random.random((dimension, 1)) + 1j * np.random.random((dimension, 1))
    return state / np.sqrt(np.vdot(state, state))


def test_state_shape():
    for size in range(3, 30):
        inital_state = randomState(size)
        target_state = randomState(size - 2)
        propagation = RandomPropagation(size, False)
        times = np.array([1.0])
        measurement = StateTransferFidelity(
            propagation=propagation,
            initialState=inital_state,
            targetState=target_state,
            times=times,
        )

        measurement.restrictSubsystems([size], [size - 2])

        for _ in range(100):
            m = measurement.measure()
            assert 0.0 <= m <= 1.0


def test_gate_shape(randomUnitaryMatrix):
    for size in range(5, 30):
        gate = randomUnitaryMatrix(size - 2)
        propagation = RandomPropagation(size, True)
        times = np.array([1.0])
        measurement = UnitaryFidelity(propagation=propagation, gate=gate, times=times)

        measurement.restrictSubsystems([size], [size - 2])

        for _ in range(100):
            m = measurement.measure()
            assert 0.0 <= m <= 1.0
