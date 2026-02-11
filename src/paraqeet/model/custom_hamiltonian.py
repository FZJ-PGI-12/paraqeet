"""Custom Hamiltonian wrapper for H(t)."""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from jax import vmap

from paraqeet.exceptions import ConfigurationException
from paraqeet.model.differentiable_hamiltonian import DifferentiableHamiltonian
from paraqeet.quantity import Array, Quantity


class CustomHamiltonian(DifferentiableHamiltonian):
    """Custom Hamiltonian class to simulate systems using a user defined Hamitonian function..

    Here we expect a Hamiltonian function of the form `H(t, *params)`.
    Here `params` is a list of scalars (**NOT `Quantity`**).

    But, `parameters` is a list of `Quantity` that would be optimized.

    It is advised to make the Hamiltonian function vmap and jit compatible.
    Furthermore, it is advised to write the Hamiltonian function in a way such that
    it takes a single time point (scalar) as input and returns  a jax array of dimensions [n, n].

    Additionally, to optimize the parameters, one needs to pass a list of
    gradient functions corresponding to each parameter, in the same order as the parameter list.

    To use open system simulation, provide a list of tuples of decay rates and corresponding collapse operators.
    """

    _hamiltonian_function: Callable[[Array, Any], Array] | Callable[[float, Any], Array]
    _parameters: list[Quantity]
    _gradient_functions: list[Callable] | None
    _collapse_operatorss: list[tuple[Array, Array]] | None

    def __init__(
        self,
        hamiltonian_function: Callable[[Array, Any], Array] | Callable[[float, Any], Array],
        parameters: list[Quantity],
        gradient_functions: list[Callable] | None = None,
        collapse_operators: list[tuple[Array, Array]] | None = None,
    ):
        self._hamiltonian_function = hamiltonian_function
        self._parameters = parameters
        # TODO: is a leading underscore missing here?
        self.gradient_functions = gradient_functions
        self.collapse_operators = collapse_operators

    @property
    def gradient_functions(self) -> list[Callable] | None:
        """Return gradient functions."""
        return self._gradient_functions

    @gradient_functions.setter
    def gradient_functions(self, grad_funcs: list[Callable] | None):
        """Set gradient functions."""
        self._gradient_functions = grad_funcs

    @property
    def collapse_operators(self) -> list[tuple[Array, Array]] | None:
        """Return collapse operators."""
        return self._collapse_operators

    @collapse_operators.setter
    def collapse_operators(self, col_ops: list[tuple[Array, Array]] | None):
        """Set collapse operators."""
        self._collapse_operators = col_ops

    def dimension(self):
        """Return dimension of the Hilbert space."""
        return self.get_value_at_timestep(jnp.array([0.0])).shape[1]

    def get_parameters(self) -> list[Quantity]:
        """Return a list of optimizable parameters."""
        return self._parameters

    def get_value_at_timestep(self, timestep: float) -> Array:
        """Return Hamiltonian as a function of time for a single time point."""
        params = [p.get_value()[0] for p in self._parameters]
        return self._hamiltonian_function(timestep, *params)

    def get_value(self, times: Array) -> Array:
        """Return Hamiltonian as a function of time for an array of time."""
        params = [p.get_value()[0] for p in self._parameters]
        matrix_fun = vmap(self._hamiltonian_function, in_axes=(0,) + (None,) * len(params))
        # ignoring mypy due to vmap
        return matrix_fun(times, *params)  # type: ignore

    def get_gradient_at_timestep(self, timestep: float) -> Array:
        """Return the gradient as a function of time for a single time point."""
        params = [p.get_value()[0] for p in self._parameters]
        if self.gradient_functions is None:
            raise ConfigurationException("Specify the gradient functions of the Hamiltonian to compute gradients.")
        if len(self.gradient_functions) != len(params):
            raise ConfigurationException(
                f"Got {len(params)} parameters but got {len(self.gradient_functions)}. "
                + "Provide gradient methods for all the input parameters"
            )
        grads = jnp.array([grad_func(timestep, *params) for grad_func in self.gradient_functions])
        return grads

    def value_and_get_gradient_at_timestep(self, time) -> tuple[Array, Array] | tuple[float, Array]:
        """Return Hamiltonian and its gradient for one timestep."""
        return self.get_value_at_timestep(time), self.get_gradient_at_timestep(time)

    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array]:
        """Return Hamiltonian and its gradient as a function of time."""
        # ignoring mypy due to vmap
        return self.get_value(times), jnp.array(vmap(self.get_gradient_at_timestep, in_axes=(0,))(times))  # type: ignore

    def get_collapseops(self) -> list[tuple[Array, Array]]:
        """Return collapse operators."""
        if self.collapse_operators is None:
            raise ConfigurationException("Collapse operators not specified.")
        return self.collapse_operators
