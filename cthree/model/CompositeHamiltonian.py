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
    __vmap: bool

    def __init__(
        self,
        subsystems: List[Hamiltonian],
        couplings: List[Coupling] | None = None,
        vmap: bool = True,
    ):
        if couplings is None:
            couplings = []
        self.__subsystems = subsystems
        self.__couplings = couplings
        self.__dimensions = [s.dimension() for s in subsystems]
        self.__totalDimension = np.prod(self.__dimensions)
        self.__vmap = vmap

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

    def setVmap(self, vmap: bool) -> None:
        self.__vmap = vmap

    def getMatrix(self, t: np.ndarray) -> np.ndarray:
        if self.__vmap:
            matrix = vmap(self._getMatrixOneTime)(t)
        else:
            matrix = self._getMatrixRepeat(t)
        return matrix

    def _getMatrixOneTime(self, t: float) -> jnp.ndarray:
        """
        Return the matrix representation of the Hamiltonian for a single time point.

        Args:
            t (float): One time step

        Returns:
            np.ndarray: Hamiltonian of shape [n, n] with n: hilbert space
        """
        # Calculate the tensor product of all subsystem matrices
        matrix = np.zeros((self.__totalDimension, self.__totalDimension))
        for n, subsystem in enumerate(self.__subsystems):
            subMatrix = subsystem.getMatrixOneTime(t)
            matrix += self.__tensorProductWithIdentity([subMatrix], [n])

        for coupling in self.__couplings:
            # Create a tensor product where all subsystems except the coupled ones are identity
            indices = [self.__subsystems.index(s) for s in coupling.getSubsystems()]
            subMatrices_terms = coupling.getMatricesOneTime(t)
            for subMatrices in subMatrices_terms:
                matrix += self.__tensorProductWithIdentity(subMatrices, indices)

        return matrix

    def _getMatrixRepeat(self, t: np.ndarray) -> jnp.ndarray:
        """
        Return the matrix representation of the Hamiltonian.

        Args:
            t (np.ndarray): Vector of time samples

        Returns:
            np.ndarray: Hamiltonian of shape [t, n, n]  with t: time, n: hilbert space
        """
        # Calculate the tensor product of all subsystem matrices
        matrix = np.zeros((len(t), self.__totalDimension, self.__totalDimension))
        for n, subsystem in enumerate(self.__subsystems):
            subMatrix = subsystem.getMatrix(t)
            for j in range(len(t)):
                matrix[j] += self.__tensorProductWithIdentity([subMatrix[j]], [n])

        for coupling in self.__couplings:
            # Create a tensor product where all subsystems except the coupled ones are identity
            indices = [self.__subsystems.index(s) for s in coupling.getSubsystems()]
            subMatrices = coupling.getMatrices(t)
            for j in range(len(t)):
                subMatricesInTime = [s[j, :, :] for s in subMatrices]
                matrix[j] += self.__tensorProductWithIdentity(
                    subMatricesInTime, indices
                )

        return matrix

    def gradient(self, t: np.ndarray) -> List[np.ndarray]:
        """
        Return the gradient of each parameter as a list.
        """
        if self.__vmap:
            grads = vmap(self._gradientOneTime)(t)
        else:
            grads = self._gradientRepeat(t)
        return grads

    def _gradientRepeat(self, t: np.ndarray) -> List[np.ndarray]:
        """
        Return the gradient of each parameter as a list.
        """
        gradients = []

        # Take the gradients from all subsystems and plug them into the tensor product with identities
        for i, subsystem in enumerate(self.__subsystems):
            subGradients = subsystem.gradient(t)
            for g in subGradients:
                gradients.append(self.__tensorProductWithIdentity([g], [i]))

        # Do the same for couplings, except that the tensor product has more than one non-identity component.
        for coupling in self.__couplings:
            indices = [self.__subsystems.index(s) for s in coupling.getSubsystems()]
            couplingGradient = coupling.gradient(t)
            for g in couplingGradient:
                gradients.append(self.__tensorProductWithIdentity(g, indices))

        return jnp.array(gradients)

    def _gradientOneTime(self, t: float) -> List[np.ndarray]:
        """
        Return the gradient of each parameter as a list.
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
        self, M: List[np.ndarray], n: List[int]
    ) -> np.ndarray:
        """
        Puts the matrices M into a tensor product at positions n where all other positions are identity matrices:

        .. math::
            1 \\otimes \\dots \\otimes 1 \\otimes M_1 \\otimes 1 \\otimes \\dots \\otimes 1 \\otimes M_2 \\dots

        The dimensions are assumed to be the same as the subsystems.
        """
        # Create identity matrices for all subsystems and fill in M at the corresponding indices
        subMatrices = [np.eye(s.dimension()) for s in self.__subsystems]
        for i, k in enumerate(n):
            subMatrices[k] = M[i]

        # Tensor product everything in subMatrices
        product = np.eye(1)
        for m in subMatrices:
            product = jnp.kron(product, m)

        return product

    def getCollapseOps(self) -> List[Tuple[float, np.ndarray]]:
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
