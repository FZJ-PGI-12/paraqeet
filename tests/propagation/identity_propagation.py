"""Test the identity propagation model."""

import jax.numpy as jnp

from paraqeet.propagation.differentiable_propagation import DifferentiablePropagation
from paraqeet.quantity import Array, Quantity


class IdentityPropagation(DifferentiablePropagation):
    """Mock identity propagation implementation.

    Returns the initial state as the target state.
    """

    _state: Array

    def __init__(self):
        super().__init__(None, None)

    def set_initial_state(self, state: Array):
        """Set the initial state of the system.

        Set it to the given state argument.

        Parameters
        ----------
        state : Array
            Given state to be set as the initial state.

        """
        self._state = state

    def propagate(self, time: Array) -> Array:
        """Get the propagated state across the timestamps.

        Parameters
        ----------
        time: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the propagated values of the state across timestamps.

        """
        return jnp.array([self._state] * len(time))

    def get_value_and_gradient(self, time: Array) -> tuple[Array, Array]:
        # Returns an empty gradient because the class has 0 parameters
        empty_gradient = jnp.zeros(shape=(len(time), 0, len(self._state)))
        return self.propagate(time), empty_gradient

    def get_parameters(self) -> list[Quantity]:
        """Returns an empty list."""
        return []
