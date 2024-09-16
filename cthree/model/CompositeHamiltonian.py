"""Class definition of the composite Hamiltonian model."""

import numpy as np
import jax.numpy as jnp
from jax import vmap

from cthree.Quantity import Quantity
from cthree.model.Coupling import Coupling
from cthree.model.Hamiltonian import Hamiltonian


class CompositeHamiltonian(Hamiltonian):
    """A hamiltonian that consists of subsystems and couplings.

    This class takes care of the tensor products.
    The list of parameters will contain the parameters of all subsystems
    and couplings in the order they were added.

    Parameters
    ----------
    subsystems : List[Hamiltonian]
        List of Hamiltonains forming the subsystems of a composite system.
    couplings: List[Coupling], optional
        List of couplings between the various subsystems
    """

    __subsystems: list[Hamiltonian]
    __couplings: list[Coupling]
    __dimensions: list[int]
    __totalDimension: int

    def __init__(
        self,
        subsystems: list[Hamiltonian],
        couplings: list[Coupling] | None = None,
    ):
        super().__init__()
        if couplings is None:
            couplings = []
        self.__subsystems = subsystems
        self.__couplings = couplings
        self.__dimensions = [s.dimension() for s in subsystems]
        self.__totalDimension = np.prod(self.__dimensions)

    def getParameters(self) -> list[Quantity]:
        """Collect parameters from all subsystems and couplings.

        Parameters
        ----------
        list[cthree.Quantity]
            Returns the list of parameters of the system.

        """
        params = []
        for subsystem in self.__subsystems:
            params += subsystem.getParameters()
        for coupling in self.__couplings:
            params += coupling.getParameters()
        return params

    def setOptimisableParameters(self, params: list[Quantity]) -> None:
        """Set optimisable parameters for the system.

        Forward parameters to the subsystems and couplings.
        All of them should find their own parameters in the list.

        Parameters
        ----------
        params : list[cthree.Quantity]
            Input list of parameters to be set.

        """
        for subsystem in self.__subsystems:
            subsystem.setOptimisableParameters(params)
        for coupling in self.__couplings:
            coupling.setOptimisableParameters(params)

    def dimension(self) -> int:
        """Return the dimension of the system.

        Returns
        -------
        int
            Dimension of the system.

        """
        return self.__totalDimension

    def getMatrixOneTime(self, t: float) -> jnp.ndarray:
        """Get matrix representation of the Hamiltonian for a single time point.

        Parameters
        ----------
        t : float
            One time step.

        Returns
        -------
        jax.numpy.ndarray
            Hamiltonian of shape [n, n] with 'n' as the Hilbert space
            dimension.

        """
        # Calculate the tensor product of all subsystem matrices
        matrix = jnp.zeros((self.__totalDimension, self.__totalDimension))
        for n, subsystem in enumerate(self.__subsystems):
            subMatrix = subsystem.getMatrixOneTime(t)
            matrix += self.__tensorProductWithIdentity([subMatrix], [n])

        for coupling in self.__couplings:
            # Create a tensor product where all subsystems
            # except the coupled ones are identity
            indices = [
                self.__subsystems.index(s) for s in coupling.getSubsystems()
            ]
            subMatrices = coupling.getMatricesOneTime(t)
            for term in subMatrices:
                matrix += self.__tensorProductWithIdentity(term, indices)

        return matrix

    def gradient(self, t: np.ndarray) -> jnp.ndarray:
        """Return the gradient of each parameter.

        Returns as an array for an array of input times.
        Uses `vmap` to iterate over time array to generate the gradients.

        Parameters
        ----------
        t: jnp.ndarray
            Array of time samples.

        Returns
        -------
        jnp.ndarray
            Gradient for each time point in the input array of times.

        """
        return vmap(self._gradientOneTime)(t)

    def _gradientOneTime(self, t: float) -> jnp.ndarray:
        """Return the gradient of each parameter as an array for one timestamp.

        Collects the gradients from every subsytem and coupling and constructs
        the matrix in the dimension of the composite system.

        Parameters
        ----------
        t: float
            One time point.

        Returns
        -------
        jnp.ndarray
            Gradient at time t.

        """
        gradients = []

        # Take the gradients from all subsystems and plug them into the
        # tensor product with identities
        for i, subsystem in enumerate(self.__subsystems):
            subGradients = subsystem.gradientOneTime(t)
            for g in subGradients:
                gradients.append(self.__tensorProductWithIdentity([g], [i]))

        # Do the same for couplings, except that the tensor product
        # has more than one non-identity component.
        for coupling in self.__couplings:
            indices = [
                self.__subsystems.index(s) for s in coupling.getSubsystems()
            ]
            couplingGradient = coupling.gradientOneTime(t)
            for term in couplingGradient:
                for g in term:
                    gradients.append(
                        self.__tensorProductWithIdentity(g, indices)
                    )

        return jnp.array(gradients)

    def __tensorProductWithIdentity(
        self, M: list[jnp.ndarray], n: list[int]
    ) -> jnp.ndarray:
        r"""Put the matrices M into a tensor product at positions `n`.

        All other positions are identity matrices:
        .. math::
            1 \\otimes \\dots \\otimes 1 \\otimes M_1 \\otimes 1
                \\otimes \\dots \\otimes 1 \\otimes M_2 \\dots
        The dimensions are assumed to be the same as the subsystems.

        Parameters
        ----------
        M : List[jax.numpy.ndarray]
            List of Matrices for tensor product
        n : List[int]
            List of indices for the each M_i

        Returns
        -------
        jax.numpy.ndarray
            Tensor product of M_i's with I's.

        """
        # Create identity matrices for all subsystems and
        # fill in M at the corresponding indices
        subMatrices = [jnp.eye(s.dimension()) for s in self.__subsystems]
        for i, k in enumerate(n):
            subMatrices[k] = M[i]

        # Tensor product everything in subMatrices
        product = jnp.eye(1)
        for m in subMatrices:
            product = jnp.kron(product, m)

        return product
