"""Class definition of the composite Hamiltonian model."""

import jax
import jax.numpy as jnp
import numpy as np

from paraqeet.exceptions import IncompatibleLayersException
from paraqeet.model.coupling import TwoBodyCoupling
from paraqeet.model.system import System
from paraqeet.model.utils import tensor_product_with_identity
from paraqeet.quantity import Array, Quantity


class CompositeSystem(System):
    """A hamiltonian that consists of subsystems and couplings.

    This class takes care of the tensor products.
    The list of parameters will contain the parameters of all subsystems
    and couplings in the order they were added.

    Parameters
    ----------
    subsystems : list[DifferentiableHamiltonian]
        List of DifferentiableHamiltonians forming the subsystems of a composite system.
    couplings: list[Coupling], optional
        List of couplings between the various subsystems
    """

    _subsystems: list[System]
    _couplings: list[TwoBodyCoupling]
    _dimensions: list[int]
    _total_dimension: int

    def __init__(
        self,
        subsystems: list[System],
        couplings: list[TwoBodyCoupling] | None = None,
    ):
        super().__init__()
        if couplings is None:
            couplings = []
        self._subsystems = subsystems
        self._couplings = couplings
        self._dimensions = [s.dimension() for s in subsystems]
        self._total_dimension = int(np.prod(self._dimensions))

    def get_parameters(self) -> list[Quantity]:
        """Collect parameters from all subsystems and couplings.

        Parameters
        ----------
        list[Quantity]
            Returns the list of parameters of the system.

        """
        params = []
        for subsystem in self._subsystems:
            params += subsystem.get_parameters()
        for coupling in self._couplings:
            params += coupling.get_parameters()
        return params

    def set_optimizable_parameters(self, params: list[Quantity]) -> None:
        """Set optimizable parameters for the system.

        Forward parameters to the subsystems and couplings.
        All of them should find their own parameters in the list.

        Parameters
        ----------
        params : list[Quantity]
            Input list of parameters to be set.

        """
        for subsystem in self._subsystems:
            subsystem.set_optimizable_parameters(params)
        for coupling in self._couplings:
            coupling.set_optimizable_parameters(params)

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

    def get_hamiltonian_at_timestep(self, timestep: float) -> Array:
        """Get matrix representation of the Hamiltonian for a single time point.

        Parameters
        ----------
        timestep: float
            One time step.

        Returns
        -------
        Array
            Hamiltonian of shape [n, n] with 'n' as the Hilbert space
            dimension.

        """
        # Calculate the tensor product of all subsystem matrices
        matrix = jnp.zeros((self._total_dimension, self._total_dimension))
        for n, subsystem in enumerate(self._subsystems):
            sub_matrix = subsystem.get_hamiltonian_at_timestep(timestep)
            matrix += tensor_product_with_identity([sub_matrix], [n], self._dimensions)

        for coupling in self._couplings:
            # Create a tensor product where all subsystems
            # except the coupled ones are identity
            indices = [self._subsystems.index(s) for s in coupling.subsystems]
            sub_matrices = coupling.get_couplings()
            for term in sub_matrices:
                matrix += tensor_product_with_identity(term, indices, self._dimensions)

        return matrix

    def get_hamiltonian_gradient_at_timestep(self, time: float) -> Array:
        """Return the gradient of each parameter as an array for one timestamp.

        Collects the gradients from every subsytem and coupling and constructs
        the matrix in the dimension of the composite system.

        Parameters
        ----------
        time: float
            One time point.

        Returns
        -------
        Array
            Gradient at time t.

        """
        gradients = []

        # Take the gradients from all subsystems and plug them into the
        # tensor product with identities
        for one_index, subsystem in enumerate(self._subsystems):
            # TODO: Fix typing
            # ignoring mypy due to vmap
            sub_gradients = subsystem.get_hamiltonian_gradient_at_timestep(jnp.array(time, ndmin=1))  # type: ignore
            for g in sub_gradients:
                if not isinstance(g, np.ndarray | jax.Array):
                    raise IncompatibleLayersException(f"Expected 'Array' got {type(g)} as gradient.")
                gradients.append(tensor_product_with_identity([g], [one_index], self._dimensions))

        # Do the same for couplings, except that the tensor product
        # has more than one non-identity component.
        for coupling in self._couplings:
            indices = [self._subsystems.index(s) for s in coupling.subsystems]
            coupling_gradient = coupling.get_coupling_gradients()
            for term in coupling_gradient:
                for g_list in term:
                    grad = tensor_product_with_identity(g_list, indices, self._dimensions)
                    if grad.size != 0:
                        gradients.append(grad)

        return jnp.array(gradients)
