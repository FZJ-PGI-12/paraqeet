from typing import List, Tuple

import numpy as np
import jax.numpy as jnp
from jax import vmap

from cthree.Quantity import Quantity
from cthree.model.Coupling import Coupling
from cthree.model.Hamiltonian import Hamiltonian


class CompositeHamiltonian(Hamiltonian):
    """
    A hamiltonian that consists of subsystems and couplings. This class takes care of the tensor products. The list
    of parameters will contain the parameters of all subsystems and couplings in the order they were added.
    """

    __subsystems: List[Hamiltonian]
    __couplings: List[Coupling]
    __dimensions: List[int]
    __totalDimension: int

    def __init__(self, subsystems: List[Hamiltonian], couplings: List[Coupling] | None = None):
        super().__init__()
        if couplings is None:
            couplings = []
        self.__subsystems = subsystems
        self.__couplings = couplings
        self.__dimensions = [s.dimension() for s in subsystems]
        self.__totalDimension = np.prod(self.__dimensions)

    def getParameters(self) -> List[Quantity]:
        # Collect parameters from all subsystems and couplings
        params = []
        for subsystem in self.__subsystems:
            params += subsystem.getParameters()
        for coupling in self.__couplings:
            params += coupling.getParameters()
        return params

    def setOptimisableParameters(self, params: List[Quantity]) -> None:
        # Forward parameters to the subsystems and couplings. All of them should find their own parameters in the list.
        for subsystem in self.__subsystems:
            subsystem.setOptimisableParameters(params)
        for coupling in self.__couplings:
            coupling.setOptimisableParameters(params)

    def dimension(self) -> int:
        return self.__totalDimension

    def getMatrixOneTime(self, t: float) -> jnp.ndarray:
        """
        Return the matrix representation of the Hamiltonian for a single time point.

        Args:
            t (float): One time step

        Returns:
            np.ndarray: Hamiltonian of shape [n, n] with n: hilbert space
        """
        # Calculate the tensor product of all subsystem matrices
        matrix = jnp.zeros((self.__totalDimension, self.__totalDimension))
        for n, subsystem in enumerate(self.__subsystems):
            subMatrix = subsystem.getMatrixOneTime(t)
            matrix += self.__tensorProductWithIdentity([subMatrix], [n])

        for coupling in self.__couplings:
            # Create a tensor product where all subsystems except the coupled ones are identity
            indices = [self.__subsystems.index(s) for s in coupling.getSubsystems()]
            subMatrices = coupling.getMatricesOneTime(t)
            matrix += self.__tensorProductWithIdentity(subMatrices, indices)

        return matrix

    def gradient(self, t: np.ndarray) -> jnp.ndarray:
        """
        Return the gradient of each parameter as a list.
        """
        return vmap(self._gradientOneTime)(t)

    def _gradientOneTime(self, t: float) -> jnp.ndarray:
        """
        Return the gradient of each parameter as a list for one timestamp.
        """
        gradients = []

        # Take the gradients from all subsystems and plug them into the tensor product with identities
        for i, subsystem in enumerate(self.__subsystems):
            subGradients = subsystem.gradientOneTime(t)
            for g in subGradients:
                gradients.append(self.__tensorProductWithIdentity([g], [i]))

        # Do the same for couplings, except that the tensor product has more than one non-identity component.
        for coupling in self.__couplings:
            indices = [self.__subsystems.index(s) for s in coupling.getSubsystems()]
            couplingGradient = coupling.gradientOneTime(t)
            for g in couplingGradient:
                gradients.append(self.__tensorProductWithIdentity(g, indices))

        return jnp.array(gradients)

    def __tensorProductWithIdentity(
        self, M: List[jnp.ndarray], n: List[int]
    ) -> jnp.ndarray:
        """
        Puts the matrices M into a tensor product at positions n where all other positions are identity matrices:

        .. math::
            1 \\otimes \\dots \\otimes 1 \\otimes M_1 \\otimes 1 \\otimes \\dots \\otimes 1 \\otimes M_2 \\dots

        The dimensions are assumed to be the same as the subsystems.
        """
        # Create identity matrices for all subsystems and fill in M at the corresponding indices
        subMatrices = [jnp.eye(s.dimension()) for s in self.__subsystems]
        for i, k in enumerate(n):
            subMatrices[k] = M[i]

        # Tensor product everything in subMatrices
        product = jnp.eye(1)
        for m in subMatrices:
            product = jnp.kron(product, m)

        return product

    def getCollapseOps(self) -> List[Tuple[float, jnp.ndarray]]:
        """
        Gather collapse operators from the subsystems and then tensor product them
        with identity to create the collapse operators of the right dimension.
        """
        allCollapseOps = []
        for n, subsystem in enumerate(self.__subsystems):
            rates_and_cols = subsystem.getCollapseOps()
            for rate, colOp in rates_and_cols:
                allCollapseOps.append(
                    (rate, self.__tensorProductWithIdentity([colOp], [n]))
                )
        return allCollapseOps
