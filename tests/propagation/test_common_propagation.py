# Helper functions that are common tests for all propagation implementations
import numpy as np

from paraqeet.propagation.propagation import Propagation


def check_propagation(propagation: Propagation, dimension: int):
    random_time_vector = np.linspace(0.0, np.random.randint(1, 10) * np.random.rand(), np.random.randint(2, 10))
    state = np.random.random(dimension) + 1j * np.random.random(dimension)
    state = state / np.sqrt(np.vdot(state, state))
    state = np.expand_dims(state, axis=1)
    propagation.initial_state = state
    propagation.get_value(random_time_vector)
