"""Test the identity propagation model."""

import jax.numpy as jnp
from paraqeet.quantity import Array
from paraqeet.propagation.state_propagation import StatePropagation
from paraqeet.quantity import Quantity


class IdentityPropagation(StatePropagation):
    """Mock identity propagation implementation.

    Returns the initial state as the target state.
    """

    __state: Array

    def __init__(self):
        super().__init__(None)

    def set_initial_state(self, state: Array):
        """Set the initial state of the system.

        Set it to the given state argument.

        Parameters
        ----------
        state : Array
            Given state to be set as the initial state.

        """
        self.__state = state

    def propagate(self, time: Array) -> Array:
        """Get the propagated state across the timestamps.

        Parameters
        ----------
        time : Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the propagated values of the state across timestamps.

        """
        return jnp.array([self.__state] * len(time))

    def get_parameters(self) -> list[Quantity]:
        """Returns an empty list."""
        return []
