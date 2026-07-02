"""Class definition of the composite Hamiltonian model."""

# AC: currently the handling of Drive is problematic. One cannot
# add drives to the composite system since they
# are automatically assumed to be None, even if they
# are in the subsystems.

from typing import override

import jax.numpy as jnp
import numpy as np

from paraqeet.model.coupling import Coupling
from paraqeet.model.hamiltonian import Hamiltonian
from paraqeet.model.utils import tensor_product_with_identity
from paraqeet.quantity import Array, Quantity


class CompositeHamiltonian(Hamiltonian):
    """A hamiltonian that consists of subsystems and couplings.

    This class takes care of the tensor products.
    The list of parameters will contain the parameters of all subsystems
    and couplings in the order they were added.

    Attributes
    ----------
    sub_hamiltonians : list[Hamiltonian]
        List of subsystems' Hamiltonians forming the a composite system.
    couplings: list[Coupling], optional
        List of couplings between the various subsystems
    """

    _sub_hamiltonians: list[Hamiltonian]
    _couplings: list[Coupling]
    _dimensions: list[int]
    _total_dimension: int

    def __init__(
        self,
        sub_hamiltonians: list[Hamiltonian],
        couplings: list[Coupling] | None = None,
    ):
        super().__init__()
        if couplings is None:
            couplings = []
        self._sub_hamiltonians = sub_hamiltonians
        self._couplings = couplings
        self._dimensions = [s.dimension() for s in sub_hamiltonians]
        self._total_dimension = int(np.prod(self._dimensions))

    @override
    def get_parameters(self) -> list[Quantity]:
        """Collect parameters from all subsystems and couplings.

        Returns
        ----------
        list[Quantity]
            Returns the list of parameters of the system.

        """
        params = []
        for sub_hamil in self._sub_hamiltonians:
            params += sub_hamil.get_parameters()
        for coupling in self._couplings:
            params += coupling.get_parameters()
        return params

    @override
    def set_optimizable_parameters(self, params: list[Quantity]) -> None:
        """Set optimizable parameters for the system.

        Forward parameters to the subsystems and couplings.
        All of them should find their own parameters in the list.

        Parameters
        ----------
        params : list[Quantity]
            Input list of parameters to be set.

        """
        for sub_hamil in self._sub_hamiltonians:
            sub_hamil.set_optimizable_parameters(params)
        for coupling in self._couplings:
            coupling.set_optimizable_parameters(params)

    @override
    def dimension(self) -> int:
        """Return the dimension of the system.

        Returns
        -------
        int
            Dimension of the system.

        """
        return self._total_dimension

    def get_subsystem_dimensions(self) -> list[int]:
        """Return a list of dimensions of each subsystem in the composite Hamiltonian.

        Returns
        -------
        list[int]
            list of dimension of each subsystem.
        """
        return self._dimensions

    @override
    def get_value(self, times: Array) -> Array:
        # Calculate the tensor product of all subsystem matrices
        matrix = jnp.zeros((self._total_dimension, self._total_dimension))
        for n, sub_hamil in enumerate(self._sub_hamiltonians):
            sub_matrix = sub_hamil.get_value(times)
            matrix += tensor_product_with_identity([sub_matrix], [n], self._dimensions)

        for coupling in self._couplings:
            matrix += coupling.get_value(times)
        return matrix

    @override
    def get_gradient(self, times: Array) -> Array:
        """Calculate the gradient of the composite Hamiltonian.

        Parameters
        ----------
        times: Array
            Array of times.

        Returns:
        ----------
            The gradient of shape (n_times, n_params, dimension, dimension) with
            n_times the number of times, n_params the number of optimizable parameters
            and dimension the dimension of the Hilbert space of the composite system.
            The parameters are ordered according to the order of the subsystems, followed
            by the parameters of the couplings with the corresponding order.
        """
        gradient = jnp.empty((*times.shape, 0, self._total_dimension, self._total_dimension))

        # Take the gradients from all subsystems and plug them into the
        # tensor product with identities
        for one_index, sub_hamil in enumerate(self._sub_hamiltonians):
            # ignoring mypy due to vmap
            sub_gradient = sub_hamil.get_gradient(times)  # type: ignore
            sub_gradient = tensor_product_with_identity([sub_gradient], [one_index], self._dimensions)
            gradient = jnp.append(gradient, sub_gradient, axis=1)
        for coupling in self._couplings:
            coup_gradient = coupling.get_gradient(times)
            gradient = jnp.append(gradient, coup_gradient, axis=1)
        return gradient
