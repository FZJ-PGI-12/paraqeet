"""Test the random propagation model."""

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit
from scipy.stats import unitary_group

from paraqeet.differentiable import Differentiable
from paraqeet.propagation.propagation import Propagation
from paraqeet.quantity import Array, Quantity
from tests.model.dummy_model import DummyEquationsOfMotion
from tests.model.empty_hamiltonian import EmptyHamiltonian


class RandomPropagation(Propagation, Differentiable):
    """Mock random propagation implementation.

    Returns random state vectors, density matrices, or propagators.

    Parameters
    ----------
    dimension: int
        Hilbert space size for the generated states.
    generate_matrices : bool, default=False
        Whether to generate matrices instead of vectors.
    auto_update: bool, default=True
        Whether to return a new random state at every call of propagate.
        If false, propagate will return the same state until update was called.
    """

    _dimension: int
    _create_matrices: bool
    _auto_update: bool
    _state: Array

    def __init__(
        self,
        dimension: int,
        generate_matrices: bool = False,
        auto_update: bool = True,
    ):
        super().__init__(DummyEquationsOfMotion(EmptyHamiltonian(0)), 1e9)
        self._dimension = dimension
        self._create_matrices = generate_matrices
        self._auto_update = auto_update
        self.update()
        self.is_open = False

    def get_parameters(self) -> list[Quantity]:
        """Returns an empty list."""
        return []

    def set_initial_state(self, state: Array):
        """Set the initial state of the system.

        Set it to the given state.

        Parameters
        ----------
        state: Array
            Given state to set as the initial state.

        """
        pass

    def propagate(self, time: Array) -> Array:
        """Propagate the system through time.

        Parameters
        ----------
        time: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the updated state of the system.

        """
        if self._auto_update:
            self.update()
        return jnp.array([self._state] * len(time))

    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array]:
        # Returns an empty gradient because the class has 0 parameters
        empty_gradient = jnp.zeros(shape=(len(times), 0, len(self._state)))
        return self.propagate(times), empty_gradient

    @staticmethod
    @partial(jit, static_argnums=(0,))
    def __create_random_dm(dim: int, rho: Array):
        rho /= jnp.trace(rho)
        u = unitary_group.rvs(dim)
        return jnp.conjugate(u.T) @ rho @ u

    @staticmethod
    @jit
    def __create_random_vec(state: Array):
        return state / jnp.sqrt(jnp.vdot(state, state))

    def update(self) -> None:
        """Update the state on propagation.

        Makes sure that the next call to propagate will return a
        new random state.

        """
        if self._create_matrices:
            # generate a random density matrix by rotating a
            # random diagonal matrix
            rho = jnp.diag(np.random.random(self._dimension))
            self._state = RandomPropagation.__create_random_dm(self._dimension, rho)
        else:
            # generate a random state vector
            state = np.random.random((self._dimension, 1)) + 1j * np.random.random((self._dimension, 1))
            self._state = RandomPropagation.__create_random_vec(state)
