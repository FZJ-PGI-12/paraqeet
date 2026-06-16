"""Test the identity propagation model."""

from typing import override

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

    @override
    def propagate(self, times: Array) -> Array:
        """Get the propagated state across the timestamps.

        Parameters
        ----------
        times: Array
            Array of times.

        Returns
        -------
        Array
            Returns the propagated values of the state across timestamps.

        """
        return jnp.array([self._state] * len(times))

    @override
    def get_value(self, times: Array) -> Array:
        return self.propagate(times)

    @override
    def get_gradient(self, times: Array) -> Array:
        # Returns an empty gradient because the class has 0 parameters
        empty_gradient = jnp.zeros(shape=(len(times), 0, len(self._state)))
        return empty_gradient

    def get_parameters(self) -> list[Quantity]:
        """Returns an empty list."""
        return []
