"""Custom Hamiltonian wrapper for H(t)."""

from collections.abc import Callable
from cthree.model.hamiltonian import Hamiltonian
from cthree.quantity import Array, Quantity
from jax import jacrev, vmap


class CustomHamiltonian(Hamiltonian):
    """Custom Hamiltonian class to simulate systems using a user defined Hamitonian function..

    Here we expect a Hamiltonian function of the form `H(t, *parameters)`.
    It is advised to make the Hamiltonian function vmap and jit compatible.
    Furthermore, it is advised to write the Hamiltonian function in a way such that
    it takes a single time point (scalar) as input and returns  a jax array of dimensions [n, n].

    `parameters` is a list of Quantites that would be optimised.

    Additionally, to optimise the parameters, one can either pass a gradient fuction.
    In case a gradient function is not provided, the fallback implementation uses automatic differentiation.
    To use automatic differentiation, one has to make sure that
    the Hamiltonian function is compatibale with `jax.jacrev`.

    To use open system simulation, provide a list of tuples of decay rates and corresponding collapse opearators.
    """

    __hamiltonian_function: Callable
    __parameters: list[Quantity]
    __gradient_function: Callable | None
    __collapse_operators: list[tuple[Array, Array]] | None

    def __init__(
        self,
        hamiltonian_function: Callable,
        parameters: list[Quantity],
        gradient_function: Callable | None = None,
        collapse_operators: list[tuple[Array, Array]] | None = None,
    ):
        self.__hamiltonian_function = hamiltonian_function
        self.__parameters = parameters
        self.__gradient_function = gradient_function
        self.__collapse_operators = collapse_operators

    def get_parameters(self) -> list[Quantity]:
        """Return a list of optimisable parameters."""
        return self.__parameters

    def get_matrix_one_time(self, t):
        """Return Hamiltonian as a function of time for a single time point."""
        return self.__hamiltonian_function(t, *self.__parameters)

    def get_matrix(self, t: Array) -> Array:
        """Return Hamiltonian as a function of time for an array of time."""
        return vmap(self.__hamiltonian_function)(t, *self.__parameters)

    def gradient_one_time(self, t):
        """Return the gradient as a function of time for a single time point."""
        params = self.__parameters
        if self.__gradient_function is None:
            argnums = tuple(i + 1 for i in range(len(self.__parameters)))
            grads = jacrev(self.__hamiltonian_function, argnums=argnums)(t, *params)
        else:
            grads = self.__gradient_function(t, *params)
        return grads

    def gradient(self, t: Array) -> Array:
        """Return Hamiltonian as a function of time for a single time point."""
        return vmap(self.gradient_one_time)

    def get_collapseops(self):
        """Return collapse operators."""
        return self.__collapse_operators
