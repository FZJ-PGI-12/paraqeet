"""Class definition of a Drive optimisable model."""

from abc import ABC

from jax import vmap
from paraqeet.quantity import Array

from paraqeet.optimisable import Optimisable


class Drive(Optimisable, ABC):
    """Represents a time-dependent drive on a subsystem.

    This can for example be a microwave or flux drive.

    """

    def get_matrix(self, annihilation_operator: Array, t: Array) -> Array:
        """Return the matrix representation of the drive.

        The dimension is given by the Hamiltonian to which this drive is
        attached. The default implementation calls getMatrixOneTime for each
        time step. Subclasses can override this function for a more efficient
        implementation.

        Parameters
        ----------
        annihilation_operator : chtree.quantity.Array
            Operator of the subsystem to which this drive is attached
        t : chtree.quantity.Array
            Vector of time samples.

        Returns
        -------
        chtree.quantity.Array
            Matrix of shape [t, n, n]  with 't' as time and 'n' as the Hilbert
            space dimension.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        return vmap(self.get_matrix_one_time, in_axes=(None, 0))(annihilation_operator, t)

    def get_matrix_one_time(self, annihilation_operator: Array, t: Array) -> Array:
        """Return the matrix representation of the drive.

        The dimension is given by the Hamiltonian to which this drive is
        attached.

        Parameters
        ----------
        annihilation_operator
            Operator of the subsystem to which this drive is attached.
        t : float
            One time point.

        Returns
        -------
        Array
            Matrix of shape [n, n]  with `n` as the Hilbert space dimension.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    def gradient(self, annihilation_operator: Array, t: Array) -> Array:
        """Return the gradient of the system.

        Returns the gradient of the matrix representation of the Hamiltonian
        with respect to each parameter as a list.

        Parameters
        ----------
        annihilation_operator : chtree.quantity.Array
            Operator of the subsystem to which this drive is attached.
        t : chtree.quantity.Array
            Vector of time samples.

        Returns
        -------
        chtree.quantity.Array
            Array of shape [t, p, n, n] with 't' as time, 'p' as number of
            parameters and 'n' as the Hilbert space dimension.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        return vmap(self.gradient_one_time, in_axes=(None, 0))(annihilation_operator, t)

    def gradient_one_time(self, annihilation_operator: Array, t: Array) -> Array:
        """Get the one-time gradient of the system.

        Returns the gradient of the matrix representation of the
        Hamiltonian with respect to each parameter as a list.

        Parameters
        ----------
        annihilation_operator : chtree.quantity.Array
            Operator of the subsystem to which this drive is attached.
        t : float
            One time step.

        Returns
        -------
        Array
            Array of shape [p, n, n] with 'p' as the number
            of parameters and 'n' as the  Hilbert space dimension.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    @staticmethod
    def _repeat(M: Array, num: int) -> Array:
        """Repeats the matrix M for each timestep in the times array.

        Returns an array with shape [t, n, m] where 't' is the
        number of time steps and 'M' is an 'n' times 'm' matrix.

        Parameters
        ----------
        M : chtree.quantity.Array
            Input matrix for repetition.
        num : int
            Number of times of repetition.

        Returns
        -------
        chtree.quantity.Array
            Repeated matrix for further computation.

        """
        return M.reshape((1,) + M.shape).repeat(num, axis=0)
