"""Class definition for a matrix representation of a Hamiltonian."""

from abc import abstractmethod

import jax.numpy as jnp
from jax import vmap

from paraqeet.model.drive import Drive
from paraqeet.optimizable import Optimizable
from paraqeet.quantity import Array, Quantity


class Hamiltonian(Optimizable):
    """Class definition for a matrix representation of a Hamiltonian.

    Implementations can contain subsystems, couplings, and drive lines
    and have to take care of frame transformations. Derived classes need to
    implement the functions getMatrix, gradient, and dimension.

    Parameters
    ----------
    drives : list[Drive]
        List of time-dependent drives.

    """

    _drives: list[Drive]

    def __init__(self, drives: list[Drive] | None = None):
        self._drives = [d for d in drives if d is not None] if drives else []

    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the Hilbert space of this Hamiltonian.

        Returns
        -------
        int
            Returns the dimension of the Hilbert space of this Hamiltonian.

        """
        pass

    def get_matrix(self, times: Array) -> Array:
        """Return the matrix representation of the Hamiltonian.

        The default implementation calls getMatrixOneTime for each time step.
        Subclasses can override this function for a more efficient
        implementation.

        Parameters
        ----------
        times: Array
            Vector of time samples.

        Returns
        -------
        Array
            Hamiltonian of shape [t, n, n]  with 't' as time and 'n' as the
            Hilbert space dimension.

        """
        return jnp.array(vmap(self.get_matrix_at_timestep)(times))

    @abstractmethod
    def get_matrix_at_timestep(self, timestep: float) -> Array:
        """Return the matrix representation of the Hamiltonian.

        Parameters
        ----------
        timestep: float
            One time point.

        Returns
        -------
        Array
            Hamiltonian of shape [n, n]  with `n` as the Hilbert space
            dimension.

        """
        pass

    @property
    def drives(self) -> list[Drive]:
        """Return the list of Drives of the system.

        Returns
        -------
        list[Drive]
            Returns a list of time-dependent drives of the system.
        """
        return self._drives

    def _get_drive_parameters(self) -> list[Quantity]:
        """Return the combined list of parameters from all drives.

        Returns
        -------
        list[Quantity]
            Returns the combined list of parameters from all drives.

        """
        params = []
        for d in self._drives:
            params += d.get_parameters()
        return params

    def _get_drive_matrix(self, annihilation_operator: Array, times: Array) -> Array:
        """Return the sum of all drives in matrix form.

        This function can be used be Hamiltonian implementations for
        including the drive. The default implementation calls
        _get_drive_matrix_one_time for each time step. Subclasses can override this
        function for a more efficient implementation.

        Parameters
        ----------
        annihilation_operator : Array
            The annihilation operator.
        times : Array
            Vector of time samples.

        Returns
        -------
        Array
            Returns the sum of all drives in matrix form.

        """
        return vmap(self._get_drive_matrix_at_timestep, in_axes=(None, 0))(annihilation_operator, times)

    def _get_drive_matrix_at_timestep(self, annihilation_operator: Array, times: float) -> Array:
        """Return the sum of all drives in matrix form.

        This function can be used be Hamiltonian implementations
        for including the drive.

        Parameters
        ----------
        annihilation_operator : Array
            The annihilation operator.
        times: Array
            Vector of time samples.

        Returns
        -------
        Array
            Returns the sum of all drives in matrix form.

        """
        dim = self.dimension()
        mat = jnp.zeros((dim, dim))
        for drive in self._drives:
            mat += drive.get_matrix_at_timestep(annihilation_operator, times)
        return mat

    def _get_drive_gradients(self, annihilation_operator: Array, times: Array) -> Array:
        """Return the gradients of all drives.

        This function can be used by Hamiltonian implementations
        for including the drive gradients.

        Parameters
        ----------
        annihilation_operator : Array
            The annihilation operator.
        times: Array
            Vector of time samples.

        Returns
        -------
        Array
            Returns the gradients of all drives.

        """
        dim = self.dimension()
        all_grads = jnp.zeros((times.shape[0], 0, dim, dim))
        for drive in self._drives:
            grads = drive.gradient(annihilation_operator, times)
            all_grads = jnp.append(all_grads, grads, axis=1)
        return all_grads

    def _get_drive_gradients_at_timestep(self, annihilation_operator: Array, time: float) -> Array:
        """Return the gradients of all drives.

        This function can be used by Hamiltonian implementations
        for including the drive gradients.

        Parameters
        ----------
        annihilation_operator : Array
            The annihilation operator.
        times: Array
            One time stamp.

        Returns
        -------
        Array
            Returns the gradients of all drives.

        """
        dim = self.dimension()
        all_grads = jnp.zeros((0, dim, dim))
        for drive in self._drives:
            grads = drive.gradient_at_timestep(annihilation_operator, time)
            all_grads = jnp.append(all_grads, grads, axis=0)
        return all_grads

    @staticmethod
    def _repeat(mat: Array, num: int) -> Array:
        """Repeat the matrix across time steps.

        Utility function that repeats the matrix mat for each timestep
        in the `num` array. Returns an array with shape [t, n, m] where
        't' is the number of time steps and 'mat' is an 'n' times 'm' matrix.

        Parameters
        ----------
        mat: Array
            Matrix for repetition.
        num: int
            Number of repetitions.

        Returns
        -------
        Array
            Repeated matrix for each time step specified.

        """
        return mat.reshape((1,) + mat.shape).repeat(num, axis=0)

    @abstractmethod
    def get_collapseops(self) -> list[tuple[Array, Array]]:
        """
        Return a list tuples of decay rates and collapse operators for each subsystem.

        Returns
        -------
        list[Tuple[Array, Array]]
            List of collapse operators
        """
        pass
