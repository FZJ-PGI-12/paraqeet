"""Class definition of the composite Hamiltonian model."""

import jax
import jax.numpy as jnp
import numpy as np

from paraqeet.exceptions import IncompatibleLayersException
from paraqeet.model.coupling import TwoBodyCoupling
from paraqeet.model.differentiable_hamiltonian import DifferentiableHamiltonian
from paraqeet.quantity import Array, Quantity


class CompositeHamiltonian(DifferentiableHamiltonian):
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

    _subsystems: list[DifferentiableHamiltonian]
    _couplings: list[TwoBodyCoupling]
    _dimensions: list[int]
    _total_dimension: int

    def __init__(
        self,
        subsystems: list[DifferentiableHamiltonian],
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

    def get_value_at_timestep(self, timestep: float) -> Array:
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
            sub_matrix = subsystem.get_value_at_timestep(timestep)
            matrix += self._tensor_product_with_identity([sub_matrix], [n])

        for coupling in self._couplings:
            # Create a tensor product where all subsystems
            # except the coupled ones are identity
            indices = [self._subsystems.index(s) for s in coupling.subsystems]
            sub_matrices = coupling.get_couplings()
            for term in sub_matrices:
                matrix += self._tensor_product_with_identity(term, indices)

        return matrix

    def get_gradient_at_timestep(self, time: float) -> Array:
        """Return the gradient of each parameter as an array for one timestamp.

        Collects the gradients from every subsystem and coupling and constructs
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
            if not isinstance(subsystem, DifferentiableHamiltonian):
                raise IncompatibleLayersException(f"Expected {subsystem} to provide gradients.")

            # TODO: Fix typing
            # ignoring mypy due to vmap
            sub_gradients = subsystem.get_gradient_at_timestep(jnp.array(time, ndmin=1))  # type: ignore
            for g in sub_gradients:
                if not isinstance(g, np.ndarray | jax.Array):
                    raise IncompatibleLayersException(f"Expected 'Array' got {type(g)} as gradient.")
                gradients.append(self._tensor_product_with_identity([g], [one_index]))

        # Do the same for couplings, except that the tensor product
        # has more than one non-identity component.
        for coupling in self._couplings:
            indices = [self._subsystems.index(s) for s in coupling.subsystems]
            coupling_gradient = coupling.get_coupling_gradients()
            for term in coupling_gradient:
                for g_list in term:
                    grad = self._tensor_product_with_identity(g_list, indices)
                    if grad.size != 0:
                        gradients.append(grad)

        return jnp.array(gradients)

    def _tensor_product_with_identity(self, mat_list: list[Array], n: list[int]) -> Array:
        r"""Put the matrices mat_list into a tensor product at positions `n`.

        All other positions are identity matrices:

        .. math::
            1 \otimes \dots \otimes 1 \otimes mat_list_1 \otimes 1
                \otimes \dots \otimes 1 \otimes mat_list_2 \dots

        The dimensions are assumed to be the same as the subsystems.

        Parameters
        ----------
        mat_list : List[Array]
            List of Matrices for tensor product
        n : list[int]
            List of indices for the each mat_list_i

        Returns
        -------
        Array
            Tensor product of mat_list_i's with I's.

        """
        # Create identity matrices for all subsystems and
        # fill in mat_list at the corresponding indices
        sub_matrices = [jnp.eye(s.dimension()) for s in self._subsystems]
        for i, k in enumerate(n):
            sub_matrices[k] = jnp.array(mat_list[i])

        # Tensor product everything in sub_matrices
        product = jnp.eye(1)
        for m in sub_matrices:
            product = jnp.kron(product, m)

        return product

    def get_collapseops(self) -> list[tuple[Array, Array]]:
        """
        Gather collapse operators from the subsystems and then tensor product them
        with identity to create the collapse operators of the right dimension.
        """
        all_collapse_ops = []
        for n, subsystem in enumerate(self._subsystems):
            rates_and_cols = subsystem.get_collapseops()
            for rate, col_op in rates_and_cols:
                all_collapse_ops.append((rate, self._tensor_product_with_identity([col_op], [n])))
        return all_collapse_ops
