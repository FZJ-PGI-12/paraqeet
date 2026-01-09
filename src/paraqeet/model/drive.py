"""Class definition of a Drive optimizable model."""

from abc import abstractmethod

from jax import vmap

from paraqeet.optimizable import Optimizable
from paraqeet.quantity import Array


class Drive(Optimizable):
    """Represents a time-dependent drive on a subsystem.

    This can for example be a microwave or flux drive.

    """

    def get_value(self, annihilation_operator: Array, times: Array) -> Array:
        """Return the matrix representation of the drive.

        The dimension is given by the Hamiltonian to which this drive is
        attached. The default implementation calls getMatrixOneTime for each
        time step. Subclasses can override this function for a more efficient
        implementation.

        Parameters
        ----------
        annihilation_operator : Array
            Operator of the subsystem to which this drive is attached
        times: Array
            Vector of time samples.

        Returns
        -------
        Array
            Matrix of shape [t, n, n]  with 't' as time and 'n' as the Hilbert
            space dimension.

        """
        return vmap(self.value_at_timestep, in_axes=(None, 0))(annihilation_operator, times)

    @abstractmethod
    def value_at_timestep(self, annihilation_operator: Array, t: Array) -> Array:
        """Return the matrix representation of the drive.

        The dimension is given by the Hamiltonian to which this drive is
        attached.

        Parameters
        ----------
        annihilation_operator
            Operator of the subsystem to which this drive is attached.
        t: float
            One time point.

        Returns
        -------
        Array
            Matrix of shape [n, n]  with `n` as the Hilbert space dimension.

        """
        pass

    def get_gradient(self, annihilation_operator: Array, t: Array) -> Array:
        """Return the gradient of the system.

        Returns the gradient of the matrix representation of the Hamiltonian
        with respect to each parameter as a list.

        Parameters
        ----------
        annihilation_operator : Array
            Operator of the subsystem to which this drive is attached.
        t: Array
            Vector of time samples.

        Returns
        -------
        Array
            Array of shape [t, p, n, n] with 't' as time, 'p' as number of
            parameters and 'n' as the Hilbert space dimension.


        """
        return vmap(self.get_gradient_at_timestep, in_axes=(None, 0))(annihilation_operator, t)

    @abstractmethod
    def get_gradient_at_timestep(self, annihilation_operator: Array, timestep: float) -> Array:
        """Get the one-time gradient of the system.

        Returns the gradient of the matrix representation of the
        Hamiltonian with respect to each parameter as a list.

        Parameters
        ----------
        annihilation_operator : Array
            Operator of the subsystem to which this drive is attached.
        timestep: float
            One time step.

        Returns
        -------
        Array
            Array of shape [p, n, n] with 'p' as the number
            of parameters and 'n' as the  Hilbert space dimension.

        """
        pass

    @staticmethod
    def _repeat(mat: Array, num: int) -> Array:
        """Repeats the matrix mat for each timestep in the times array.

        Returns an array with shape [t, n, m] where 't' is the
        number of time steps and 'mat' is an 'n' times 'm' matrix.

        Parameters
        ----------
        mat: Array
            Input matrix for repetition.
        num : int
            Number of times of repetition.

        Returns
        -------
        Array
            Repeated matrix for further computation.

        """
        return mat.reshape((1,) + mat.shape).repeat(num, axis=0)
