"""Test the identity propagation model."""

import jax.numpy as jnp

from paraqeet.differentiable import Differentiable
from paraqeet.propagation.propagation import Propagation
from paraqeet.quantity import Array, Quantity


class IdentityPropagation(Propagation, Differentiable):
    """Mock identity propagation implementation.

    Returns the initial state as the target state.
    """

    _state: Array

    def __init__(self):
        super().__init__(None, 1.0, jnp.zeros((1, 1)))

    def _propagate(self, eom: Array, state: Array, steps: Array, *args, **kwargs) -> Array:
        # This mock returns the initial state instead of propagating, see get_value.
        raise NotImplementedError

    def set_initial_state(self, state: Array):
        """Set the initial state of the system.

        Set it to the given state argument.

        Args:
            state: Given state to be set as the initial state.
        """
        self._state = state

    def get_value(self, times: Array) -> Array:
        return jnp.array([self._state] * len(times))

    def get_gradient(self, times: Array) -> Array:
        # Returns an empty gradient because the class has 0 parameters
        empty_gradient = jnp.zeros(shape=(len(times), 0, len(self._state)))
        return empty_gradient

    def get_parameters(self) -> list[Quantity]:
        return []
