"""Class definition for a matrix representation of a Hamiltonian."""

from abc import abstractmethod

import jax.numpy as jnp
from jax import vmap

from paraqeet.model.drive import Drive
from paraqeet.optimizable import Optimizable
from paraqeet.quantity import Array, Quantity


class System(Optimizable):
    """Class definition for a matrix representation of a Hamiltonian.

    Implementations can contain subsystems, couplings, and drive lines
    and have to take care of frame transformations. Derived classes need to
    implement the functions get_hamiltonian, get_hamiltonian_and_gradient, dimension, get_value_at_timestep

    Parameters
    ----------
    drives : list[Drive]
        List of time-dependent drives.

    """

    drives: list[Drive]

    def __init__(self, drives: list[Drive] | None = None):
        self.drives = [d for d in drives if d is not None] if drives else []

    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the Hilbert space of this Hamiltonian.

        Returns
        -------
        int
            Returns the dimension of the Hilbert space of this Hamiltonian.

        """
        pass

    def get_hamiltonian(self, times: Array) -> Array:
        """Return the matrix representation of the Hamiltonian.

        The default implementation calls get_value_at_timestep for each time step.
        Subclasses can override this function for a more efficient
        implementation.

        Parameters
        ----------
        times: Array
            Array of time samples.

        Returns
        -------
        Array
            Hamiltonian of shape [t, n, n]  with 't' as time and 'n' as the
            Hilbert space dimension.

        """
        # Ignoring mypy due to vmap
        return jnp.array(vmap(self.get_hamiltonian_at_timestep)(times))  # type: ignore

    @abstractmethod
    def get_hamiltonian_at_timestep(self, t: float) -> Array:
        """Return the matrix representation of the Hamiltonian.

        Parameters
        ----------
        t: float
            Time.

        Returns
        -------
        Array
            Hamiltonian of shape [n, n]  with `n` as the Hilbert space
            dimension.

        """
        pass

    @abstractmethod
    def get_hamiltonian_gradient_at_timestep(self, t: float) -> Array:
        """Get the one-time gradient of the Hamiltonain.

        Returns the gradient of the matrix representation of the
        Hamiltonian with respect to each parameter as a list.

        Parameters
        ----------
        t: float
            Time.

        Returns
        -------
        Array
            Gradient as an array of shape [p, n, n] with 'p' as the number
            of parameters and 'n' as the  Hilbert space dimension.

        """
        pass

    def get_hamiltonian_and_gradient(self, times: Array) -> tuple[Array, Array]:
        """Get the matrix representation and one-time gradient of the Hamiltonain.

        Returns the gradient of the matrix representation of the
        Hamiltonian with respect to each parameter as a list.

        Parameters
        ----------
        t: float
            Time.

        Returns
        -------
        tuple[Array, Array]
            A tuple with Hamiltonian of shape [n, n]  with `n` as the Hilbert
            space dimension and gradient as and array of shape [p, n, n]
            with 'p' as the number of parameters and 'n' as the
            Hilbert space dimension.

        """
        # ignoring mypy due to vmap
        hamil_and_grad = (self.get_hamiltonian(times), vmap(self.get_hamiltonian_gradient_at_timestep)(times))  # type: ignore
        return hamil_and_grad

    def get_drive_parameters(self) -> list[Quantity]:
        """Return the combined list of parameters from all drives.

        Returns
        -------
        list[Quantity]
            Returns the combined list of parameters from all drives.

        """
        params = []
        for d in self.drives:
            params += d.get_parameters()
        return params

    def get_drive_matrix(self, times: Array) -> Array:
        """Return the sum of all drives in matrix form.

        This function can be used be Hamiltonian implementations for
        including the drive. The default implementation calls
        _get_drive_matrix_one_time for each time step. Subclasses can override this
        function for a more efficient implementation.

        Parameters
        ----------
        times: Array
            Array of time samples.

        Returns
        -------
        Array
            Returns the sum of all drives in matrix form.

        """
        # ignoring mypy due to vmap
        return vmap(self.get_drive_matrix_at_timestep, in_axes=(None, 0))(times)  # type: ignore

    def get_drive_matrix_at_timestep(self, t: float) -> Array:
        """Return the sum of all drives in matrix form.

        This function can be used be Hamiltonian implementations
        for including the drive.

        Parameters
        ----------
        t: float
            Time.

        Returns
        -------
        Array
            Returns the sum of all drives in matrix form.

        """
        dim = self.dimension()
        mat = jnp.zeros((dim, dim))
        for drive in self.drives:
            mat += drive.get_value_at_timestep(t)
        return mat

    def get_drive_gradients(self, times: Array) -> Array:
        """Return the gradients of all drives.

        This function can be used by Hamiltonian implementations
        for including the drive gradients.

        Parameters
        ----------
        times: Array
            Vector of time samples.

        Returns
        -------
        Array
            Returns the gradients of all drives.

        """
        dim = self.dimension()
        all_grads = jnp.zeros((times.shape[0], 0, dim, dim))
        for drive in self.drives:
            grads = drive.get_gradient(times)
            all_grads = jnp.append(all_grads, grads, axis=1)
        return all_grads

    def get_drive_gradients_at_timestep(self, time: float) -> Array:
        """Return the gradients of all drives.

        This function can be used by Hamiltonian implementations
        for including the drive gradients.

        Parameters
        ----------
        times: Array
            Time.

        Returns
        -------
        Array
            Returns the gradients of all drives.

        """
        dim = self.dimension()
        all_grads = jnp.zeros((0, dim, dim))
        for drive in self.drives:
            grads = drive.get_gradient_at_timestep(time)
            all_grads = jnp.append(all_grads, grads, axis=0)
        return all_grads


class OpenSystem(System):
    """System description that adds jump operators for
    the simulation of dissipation, etc.
    """

    @abstractmethod
    def get_jump_operators(self) -> list[Array]:
        """
        Return a list of jump operators for each subsystem (multiplied by the sqrt of their decay rates).

        Returns
        -------
        list[Array]
            List of jump operators
        """
        pass
