import jax.numpy as jnp

from cthree.Optimisable import Optimisable


class Drive(Optimisable):
    """
    Represents a time-dependent drive on a subsystem. This can for example be a microwave or flux drive.
    """

    def getMatrix(
        self, annihilationOperator: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Return the matrix representation of the drive. The dimension is given by the Hamiltonian to which this drive
        is attached.

        Args:
            annihilationOperator: operator of the subsystem to which this drive is attached
            t (np.ndarray): Vector of time samples

        Returns:
            np.ndarray: matrix of shape [t, n, n]  with t: time, n: hilbert space dimension
        """
        raise NotImplementedError()

    def getMatrixOneTime(
        self, annihilationOperator: jnp.ndarray, t: float
    ) -> jnp.ndarray:
        """
        Return the matrix representation of the drive. The dimension is given by the Hamiltonian to which this drive
        is attached.

        Args:
            annihilationOperator: operator of the subsystem to which this drive is attached
            t (float): One time point

        Returns:
            np.ndarray: matrix of shape [n, n]  with n: hilbert space dimension
        """
        raise NotImplementedError()

    def gradient(
        self, annihilationOperator: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Return the gradient of the matrix representation of the Hamiltonian with respect to each parameter as a list.

        Args:
            annihilationOperator: operator of the subsystem to which this drive is attached
            t (np.ndarray): Vector of time samples

        Returns:
            np.ndarray: array of shape [t, p, n, n] with t: time, p: number of parameters, n: hilbert space dimension
        """
        raise NotImplementedError()

    def gradientOneTime(
        self, annihilationOperator: jnp.ndarray, t: float
    ) -> jnp.ndarray:
        """
        Return the gradient of the matrix representation of the Hamiltonian with respect to each parameter as a list.

        Args:
            annihilationOperator: operator of the subsystem to which this drive is attached
            t (float): One time step

        Returns:
            np.ndarray: array of shape [p, n, n] with p: number of parameters, n: hilbert space dimension
        """
        raise NotImplementedError()

    @staticmethod
    def _repeat(M: jnp.ndarray, num: int) -> jnp.ndarray:
        """
        Utility function that repeats the matrix M for each timestep in the times array. Returns an array with shape
        [t, n, m] where t is the number of time steps and M is a n times m matrix.
        """
        return M.reshape((1,) + M.shape).repeat(num, axis=0)
