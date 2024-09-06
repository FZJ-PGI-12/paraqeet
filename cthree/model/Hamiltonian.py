from typing import List

import jax.numpy as jnp
from jax import vmap

from cthree.Optimisable import Optimisable
from cthree.Quantity import Quantity
from cthree.model.Drive import Drive


class Hamiltonian(Optimisable):
    """
    Matrix representation of a Hamiltonian. Implementations can contain subsystems, couplings, and drive lines and have
    to take care of frame transformations. Derived classes need to implement the functions getMatrix, gradient, and
    dimension.
    """

    _drives: List[Drive]

    def __init__(self, drives=None):
        self._drives = [d for d in drives if d is not None] if drives else []

    def dimension(self) -> int:
        """
        Returns the dimension of the Hilbert space of this Hamiltonian.
        """
        raise NotImplementedError()

    def getMatrix(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Return the matrix representation of the Hamiltonian. The default implementation calls getMatrixOneTime for each
        time step. Subclasses can override this function for a more efficient implementation.

        Args:
            t (np.ndarray): Vector of time samples

        Returns:
            jnp.ndarray: Hamiltonian of shape [t, n, n]  with t: time, n: hilbert space dimension
        """
        return vmap(self.getMatrixOneTime)(t)

    def getMatrixOneTime(self, t: float) -> jnp.ndarray:
        """
        Return the matrix representation of the Hamiltonian.

        Args:
            t (float): One time point

        Returns:
            jnp.ndarray: Hamiltonian of shape [n, n]  with n: hilbert space dimension
        """
        raise NotImplementedError()

    def gradient(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Return the gradient of the matrix representation of the Hamiltonian with respect to each parameter for each time
        step in t. Implementations must make sure that only derivatives with respect to those parameters are included
        in the gradient that were registered in the Optimisable parent class. The order of the gradients should match
        the order of the parameters returned by getParameters.

        The default implementation calls gradientOneTime for each time step. Subclasses can override this function for a
        more efficient implementation.

        Args:
            t (np.ndarray): Vector of time samples

        Returns:
            jnp.ndarray: Hamiltonian of shape [t, p, n, n]  with t: time, p: number of parameters, n: hilbert space
                        dimension
        """
        return vmap(self.gradientOneTime)(t)

    def gradientOneTime(self, t: float) -> jnp.ndarray:
        """
        Return the gradient of the matrix representation of the Hamiltonian with respect to each parameter for one time
        step t. Implementations must make sure that only derivatives with respect to those parameters are included
        in the gradient that were registered in the Optimisable parent class. The order of the gradients should match
        the order of the parameters returned by getParameters.

        Args:
            t (float): one time step

        Returns:
            jnp.ndarray: Hamiltonian of shape [p, n, n]  with p: number of parameters, n: hilbert space
                        dimension
        """
        raise NotImplementedError()

    def getDrives(self) -> List[Drive]:
        return self._drives

    def _getDriveParameters(self) -> List[Quantity]:
        """
        Returns the combined list of parameters from all drives.
        """
        params = []
        for d in self._drives:
            params += d.getParameters()
        return params

    def _getDriveMatrix(
        self, annihilationOperator: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Returns the sum of all drives in matrix form. This function can be used be Hamiltonian implementations for
        including the drive.

        The default implementation calls _getDriveMatrixOneTime for each time step. Subclasses can override this
        function for a more efficient implementation.
        """
        return vmap(self._getDriveMatrixOneTime, in_axes=(None, 0))(
            annihilationOperator, t
        )

    def _getDriveMatrixOneTime(
        self, annihilationOperator: jnp.ndarray, t: float
    ) -> jnp.ndarray:
        """
        Returns the sum of all drives in matrix form. This function can be used be Hamiltonian implementations for
        including the drive.
        """
        dim = self.dimension()
        M = jnp.zeros((dim, dim))
        for drive in self._drives:
            M += drive.getMatrixOneTime(annihilationOperator, t)
        return M

    def _getDriveGradients(
        self, annihilationOperator: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Returns the gradients of all drives. This function can be used be Hamiltonian implementations for including
        the drive gradients.
        """
        dim = self.dimension()
        allGrads = jnp.zeros((t.shape[0], 0, dim, dim))
        for drive in self._drives:
            grads = drive.gradient(annihilationOperator, t)
            allGrads = jnp.append(allGrads, grads, axis=1)
        return allGrads

    def _getDriveGradientsOneTime(
        self, annihilationOperator: jnp.ndarray, t: float
    ) -> jnp.ndarray:
        """
        Returns the gradients of all drives. This function can be used by Hamiltonian implementations for including
        the drive gradients.
        """
        dim = self.dimension()
        allGrads = jnp.zeros((0, dim, dim))
        for drive in self._drives:
            grads = drive.gradientOneTime(annihilationOperator, t)
            allGrads = jnp.append(allGrads, grads, axis=0)
        return allGrads

    @staticmethod
    def _repeat(M: jnp.ndarray, num: int) -> jnp.ndarray:
        """
        Utility function that repeats the matrix M for each timestep in the times array. Returns an array with shape
        [t, n, m] where t is the number of time steps and M is a n times m matrix.
        """
        return M.reshape((1,) + M.shape).repeat(num, axis=0)
