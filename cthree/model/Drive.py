"""Class definition of a Drive optimisable model."""

import jax.numpy as jnp
from jax import vmap

from cthree.Optimisable import Optimisable


class Drive(Optimisable):
    """Represents a time-dependent drive on a subsystem.

    This can for example be a microwave or flux drive.

    """

    def getMatrix(
        self, annihilationOperator: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        """Return the matrix representation of the drive.

        The dimension is given by the Hamiltonian to which this drive is
        attached. The default implementation calls getMatrixOneTime for each
        time step. Subclasses can override this function for a more efficient
        implementation.

        Parameters
        ----------
        annihilationOperator : jax.numpy.ndarray
            Operator of the subsystem to which this drive is attached
        t : jax.numpy.ndarray
            Vector of time samples.

        Returns
        -------
        jax.numpy.ndarray
            Matrix of shape [t, n, n]  with 't' as time and 'n' as the Hilbert
            space dimension.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        return vmap(self.getMatrixOneTime, in_axes=(None, 0))(
            annihilationOperator, t
        )

    def getMatrixOneTime(
        self, annihilationOperator: jnp.ndarray, t: float
    ) -> jnp.ndarray:
        """Return the matrix representation of the drive.

        The dimension is given by the Hamiltonian to which this drive is
        attached.

        Parameters
        ----------
        annihilationOperator
            Operator of the subsystem to which this drive is attached.
        t : float
            One time point.

        Returns
        -------
        np.ndarray
            Matrix of shape [n, n]  with `n` as the Hilbert space dimension.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    def gradient(
        self, annihilationOperator: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        """Return the gradient of the system.

        Returns the gradient of the matrix representation of the Hamiltonian
        with respect to each parameter as a list.

        Parameters
        ----------
        annihilationOperator : jax.numpy.ndarray
            Operator of the subsystem to which this drive is attached.
        t : jax.numpy.ndarray
            Vector of time samples.

        Returns
        -------
        jax.numpy.ndarray
            Array of shape [t, p, n, n] with 't' as time, 'p' as number of
            parameters and 'n' as the Hilbert space dimension.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        return vmap(self.gradientOneTime, in_axes=(None, 0))(
            annihilationOperator, t
        )

    def gradientOneTime(
        self, annihilationOperator: jnp.ndarray, t: float
    ) -> jnp.ndarray:
        """Get the one-time gradient of the system.

        Returns the gradient of the matrix representation of the
        Hamiltonian with respect to each parameter as a list.

        Parameters
        ----------
        annihilationOperator : jax.numpy.ndarray
            Operator of the subsystem to which this drive is attached.
        t : float
            One time step.

        Returns
        -------
        np.ndarray
            Array of shape [p, n, n] with 'p' as the number
            of parameters and 'n' as the  Hilbert space dimension.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    @staticmethod
    def _repeat(M: jnp.ndarray, num: int) -> jnp.ndarray:
        """Repeats the matrix M for each timestep in the times array.

        Returns an array with shape [t, n, m] where 't' is the
        number of time steps and 'M' is an 'n' times 'm' matrix.

        Parameters
        ----------
        M : jax.numpy.ndarray
            Input matrix for repetition.
        num : int
            Number of times of repetition.

        Returns
        -------
        jax.numpy.ndarray
            Repeated matrix for further computation.

        """
        return M.reshape((1,) + M.shape).repeat(num, axis=0)
