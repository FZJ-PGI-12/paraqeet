"""Custom Hamiltonian wrapper for H(t)."""

from collections.abc import Callable
from typing import Any
from paraqeet.exceptions import ConfigurationException
from paraqeet.model.hamiltonian import Hamiltonian
from paraqeet.quantity import Array, Quantity

import jax.numpy as jnp
from jax import vmap


class CustomHamiltonian(Hamiltonian):
    """Custom Hamiltonian class to simulate systems using a user defined Hamitonian function..

    Here we expect a Hamiltonian function of the form `H(t, *params)`.
    Here `params` is a list of scalars (**NOT `Quantity`**).

    But, `parameters` is a list of `Quantity` that would be optimised.

    It is advised to make the Hamiltonian function vmap and jit compatible.
    Furthermore, it is advised to write the Hamiltonian function in a way such that
    it takes a single time point (scalar) as input and returns  a jax array of dimensions [n, n].

    Additionally, to optimise the parameters, one needs to pass a list of
    gradient functions correspoding to each parameter, in the same order as the parameter list.

    To use open system simulation, provide a list of tuples of decay rates and corresponding collapse opearators.
    """

    __hamiltonian_function: Callable[[Array, Any], Array]
    __parameters: list[Quantity]
    __gradient_functions: list[Callable] | None
    __collapse_operators: list[tuple[Array, Array]] | None

    def __init__(
        self,
        hamiltonian_function: Callable[[Array, Any], Array],
        parameters: list[Quantity],
        gradient_functions: list[Callable] | None = None,
        collapse_operators: list[tuple[Array, Array]] | None = None,
    ):
        self.__hamiltonian_function = hamiltonian_function
        self.__parameters = parameters
        self.gradient_functions = gradient_functions
        self.collapse_operators = collapse_operators

    @property
    def gradient_functions(self) -> list[Callable] | None:
        """Return gradient functions."""
        return self.__gradient_functions

    @gradient_functions.setter
    def gradient_functions(self, grad_funcs: list[Callable] | None):
        """Set gradient functions."""
        self.__gradient_functions = grad_funcs

    @property
    def collapse_operators(self) -> list[tuple[Array, Array]] | None:
        """Return collapse operators."""
        return self.__collapse_operators

    @collapse_operators.setter
    def collapse_operators(self, col_ops: list[tuple[Array, Array]] | None):
        """Set collapse operators."""
        self.__collapse_operators = col_ops

    def dimension(self):
        """Return dimension of the Hilbert space."""
        return self.get_matrix_one_time(jnp.array([0.0])).shape[1]

    def get_parameters(self) -> list[Quantity]:
        """Return a list of optimisable parameters."""
        return self.__parameters

    def get_matrix_one_time(self, t):
        """Return Hamiltonian as a function of time for a single time point."""
        params = [p.get_value()[0] for p in self.__parameters]
        return self.__hamiltonian_function(t, *params)

    def get_matrix(self, t: Array) -> Array:
        """Return Hamiltonian as a function of time for an array of time."""
        params = [p.get_value()[0] for p in self.__parameters]
        matrix_fun = vmap(self.__hamiltonian_function, in_axes=(0,) + (None,) * len(params))
        return matrix_fun(t, *params)

    def gradient_one_time(self, t):
        """Return the gradient as a function of time for a single time point."""
        params = [p.get_value()[0] for p in self.__parameters]
        if self.gradient_functions is None:
            raise ConfigurationException("Specify the gradient functions of the Hamiltonian to compute gradients.")
        if len(self.gradient_functions) != len(params):
            raise ConfigurationException(
                f"Got {len(params)} parameters but got {len(self.gradient_functions)}. "
                + "Provide gradient methods for all the input paramters"
            )
        grads = jnp.array([grad_func(t, *params) for grad_func in self.gradient_functions])
        return grads

    def gradient(self, t: Array) -> Array:
        """Return Hamiltonian as a function of time for a single time point."""
        return jnp.array(vmap(self.gradient_one_time, in_axes=(0,))(t))

    def get_collapseops(self) -> list[tuple[Array, Array]]:
        """Return collapse operators."""
        if self.collapse_operators is None:
            raise ConfigurationException("Collapse operators not specified.")
        return self.collapse_operators
