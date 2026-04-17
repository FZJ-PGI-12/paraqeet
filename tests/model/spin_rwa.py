"""Class definition of a two level system (TLS) for testing."""

import jax.numpy as jnp

from paraqeet.model.system import System
from paraqeet.quantity import Array


class SpinRWA(System):
    """A Single Spin."""

    def __init__(self, drives=None):
        super().__init__(drives)
        self.sigma_p = jnp.array([[0j, 1], [0, 0]])
        self.dim = 2

    def get_hamiltonian_at_timestep(self, timestep: float) -> Array:
        """Just sigma-X."""
        return self._drives[0].get_value_at_timestep(self.sigma_p, timestep)

    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array]:
        """Gradient is just the drive matrix."""
        return self.get_hamiltonian(times), self._drives[0].get_gradient(self.sigma_p, times)

    def get_gradient_at_timestep(self, time):
        return self._drives[0].get_gradient_at_timestep(self.sigma_p, time)

    # TODO: implement dimension-method from Hamiltonian
    def dimension(self) -> int:
        raise NotImplementedError()

    #  TODO: implement get_collapseops method from Hamiltonian
    def get_collapseops(self) -> list[tuple[Array, Array]]:
        raise NotImplementedError()

    #  TODO: implement get_parameters method from Optimizable
    def get_parameters(self):
        raise NotImplementedError()
