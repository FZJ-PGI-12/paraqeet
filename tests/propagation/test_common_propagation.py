# Helper functions that are common tests for all propagation implementations
import numpy as np
import pytest

from paraqeet.exceptions import ConfigurationException
from paraqeet.propagation.state_propagation import StatePropagation


def needs_initial_state(propagation: StatePropagation, dimension: int):
    random_time_vector = np.linspace(0.0, np.random.randint(1, 10) * np.random.rand(), np.random.randint(2, 10))
    with pytest.raises(ConfigurationException):
        propagation.propagate(random_time_vector)

    state = np.random.random(dimension) + 1j * np.random.random(dimension)
    state = state / np.sqrt(np.vdot(state, state))
    propagation.set_initial_state(state)
    propagation.propagate(random_time_vector)
