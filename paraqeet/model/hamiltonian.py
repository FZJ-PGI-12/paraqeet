"""Class definition for a matrix representation of a Hamiltonian."""

from abc import ABC

import jax.numpy as jnp
from jax import vmap

from paraqeet.model.drive import Drive
from paraqeet.optimisable import Optimisable
from paraqeet.quantity import Quantity, Array


class Hamiltonian(Optimisable, ABC):
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

    def dimension(self) -> int:
        """Return the dimension of the Hilbert space of this Hamiltonian.

        Returns
        -------
        int
            Returns the dimension of the Hilbert space of this Hamiltonian.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    def get_matrix(self, t: Array) -> Array:
        """Return the matrix representation of the Hamiltonian.

        The default implementation calls getMatrixOneTime for each time step.
        Subclasses can override this function for a more efficient
        implementation.

        Parameters
        ----------
        t: Array
            Vector of time samples.

        Returns
        -------
        Array
            Hamiltonian of shape [t, n, n]  with 't' as time and 'n' as the
            Hilbert space dimension.

        """
        return jnp.array(vmap(self.get_matrix_one_time)(t))

    def get_matrix_one_time(self, t: Array) -> Array:
        """Return the matrix representation of the Hamiltonian.

        Parameters
        ----------
        t: Array
            One time point.

        Returns
        -------
        Array
            Hamiltonian of shape [n, n]  with `n` as the Hilbert space
            dimension.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    def gradient(self, t: Array) -> Array:
        """Return the gradient of the system.

        Returns the gradient of the matrix representation of the Hamiltonian
        with respect to each parameter for each time step in t. Implementations
        must make sure that only derivatives with respect to those parameters
        are included in the gradient that were registered in the Optimisable
        parent class. The order of the gradients should match the order of the
        parameters returned by getParameters. The default implementation calls
        gradient_one_time for each time step. Subclasses can override this
        function for a more efficient implementation.

        Parameters
        ----------
        t: Array
            Vector of time samples.

        Returns
        -------
        Array
            Hamiltonian of shape [t, p, n, n]  with 't' as time, 'p' as number
            of parameters and 'n' as Hilbert space dimension.

        """
        return vmap(self.gradient_one_time)(t)

    def gradient_one_time(self, t: Array) -> Array:
        """Return the one-time gradient of the system.

        Return the gradient of the matrix representation of the Hamiltonian
        with respect to each parameter for one time step t.
        Implementations must make sure that only derivatives with respect
        to those parameters are included in the gradient that were registered
        in the Optimisable parent class. The order of the gradients should match
        the order of the parameters returned by getParameters.

        Parameters
        ----------
        t: Array
            One time step.

        Returns
        -------
        list[Array]
            Hamiltonian of shape [p, n, n]  with 'p' as the number
            of parameters and 'n' as the Hilbert space dimension.

        """
        raise NotImplementedError()

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

    def _get_drive_matrix(self, annihilation_operator: Array, t: Array) -> Array:
        """Return the sum of all drives in matrix form.

        This function can be used be Hamiltonian implementations for
        including the drive. The default implementation calls
        _get_drive_matrix_one_time for each time step. Subclasses can override this
        function for a more efficient implementation.

        Parameters
        ----------
        annihilation_operator : paraqeet.quantity.Array
            The annihilation operator.
        t : paraqeet.quantity.Array
            Vector of time samples.

        Returns
        -------
        Array
            Returns the sum of all drives in matrix form.

        """
        return vmap(self._get_drive_matrix_one_time, in_axes=(None, 0))(annihilation_operator, t)

    def _get_drive_matrix_one_time(self, annihilation_operator: Array, t: Array) -> Array:
        """Return the sum of all drives in matrix form.

        This function can be used be Hamiltonian implementations
        for including the drive.

        Parameters
        ----------
        annihilation_operator : paraqeet.quantity.Array
            The annihilation operator.
        t: Array
            Vector of time samples.

        Returns
        -------
        Array
            Returns the sum of all drives in matrix form.

        """
        dim = self.dimension()
        mat = jnp.zeros((dim, dim))
        for drive in self._drives:
            mat += drive.get_matrix_one_time(annihilation_operator, t)
        return mat

    def _get_drive_gradients(self, annihilation_operator: Array, t: Array) -> Array:
        """Return the gradients of all drives.

        This function can be used by Hamiltonian implementations
        for including the drive gradients.

        Parameters
        ----------
        annihilation_operator : paraqeet.quantity.Array
            The annihilation operator.
        t: paraqeet.quantity.Array
            Vector of time samples.

        Returns
        -------
        Array
            Returns the gradients of all drives.

        """
        dim = self.dimension()
        all_grads = jnp.zeros((t.shape[0], 0, dim, dim))
        for drive in self._drives:
            grads = drive.gradient(annihilation_operator, t)
            all_grads = jnp.append(all_grads, grads, axis=1)
        return all_grads

    def _get_drive_gradients_one_time(self, annihilation_operator: Array, t: Array) -> Array:
        """Return the gradients of all drives.

        This function can be used by Hamiltonian implementations
        for including the drive gradients.

        Parameters
        ----------
        annihilation_operator : paraqeet.quantity.Array
            The annihilation operator.
        t: Array
            One time stamp.

        Returns
        -------
        Array
            Returns the gradients of all drives.

        """
        dim = self.dimension()
        all_grads = jnp.zeros((0, dim, dim))
        for drive in self._drives:
            grads = drive.gradient_one_time(annihilation_operator, t)
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
        mat: paraqeet.quantity.Array
            Matrix for repetition.
        num: int
            Number of repetitions.

        Returns
        -------
        Array
            Repeated matrix for each time step specified.

        """
        return mat.reshape((1,) + mat.shape).repeat(num, axis=0)

    def get_collapseops(self) -> list[tuple[Array, Array]]:
        """
        Return a list tuples of decay rates and collapse operators for each subsystem.

        Returns
        -------
        list[Tuple[Array, Array]]
            List of collapse operators
        """
        raise NotImplementedError()
