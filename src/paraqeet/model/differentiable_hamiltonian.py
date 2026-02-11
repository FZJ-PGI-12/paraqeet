from abc import abstractmethod

from jax import vmap

from paraqeet.differentiable import Differentiable
from paraqeet.model.hamiltonian import Hamiltonian
from paraqeet.quantity import Array


class DifferentiableHamiltonian(Differentiable, Hamiltonian):
    """
    A Hamiltonian that also satisfies the differentiable interface. It requires a gradient function that works on a
    per timestep basis and provides a vectorized mapping to evaluate multiple time values.
    """

    @abstractmethod
    def get_gradient_at_timestep(self, time: float) -> Array:
        """Compute the gradient of this Hamiltonian wrt to parameters for a single timestep."""
        pass

    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array]:
        """Compute value and gradient for given timesteps. The gradient call uses vmap over at_timestep methods."""
        # ignoring mypy due to vmap
        return self.get_value(times), vmap(self.get_gradient_at_timestep)(times)  # type: ignore
