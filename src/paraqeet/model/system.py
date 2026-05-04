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

    def get_hamiltonian(self, times: Array) -> Array:
        """Return the matrix representation of the Hamiltonian.

        The default implementation calls get_value_at_timestep for each time step.
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
        # Ignoring mypy due to vmap
        return jnp.array(vmap(self.get_hamiltonian_at_timestep)(times))  # type: ignore

    @abstractmethod
    def get_hamiltonian_at_timestep(self, timestep: float) -> Array:
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

    @abstractmethod
    def get_hamiltonian_gradient_at_timestep(self, time: float) -> Array:
        """Compute the gradient of this Hamiltonian wrt to parameters for a single timestep."""
        pass

    def get_hamiltonian_and_gradient(self, times: Array) -> tuple[Array, Array]:
        """Compute value and gradient for given timesteps. The gradient call uses vmap over at_timestep methods."""
        # ignoring mypy due to vmap
        return self.get_hamiltonian(times), vmap(self.get_hamiltonian_gradient_at_timestep)(times)  # type: ignore

    @property
    def drives(self) -> list[Drive]:
        """Return the list of Drives of the system.

        Returns
        -------
        list[Drive]
            Returns a list of time-dependent drives of the system.
        """
        return self._drives

    @drives.setter
    def drives(self, drives: list[Drive]) -> None:
        """Set the drives

        Parameters
        ----------
        drives: list[Drive]
            List of drives to set.
        """
        self._drives = drives

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
        # ignoring mypy due to vmap
        return vmap(self._get_drive_matrix_at_timestep, in_axes=(None, 0))(annihilation_operator, times)  # type: ignore

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
            mat += drive.get_value_at_timestep(annihilation_operator, times)
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
            grads = drive.get_gradient(annihilation_operator, times)
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
            grads = drive.get_gradient_at_timestep(annihilation_operator, time)
            all_grads = jnp.append(all_grads, grads, axis=0)
        return all_grads


class OpenSystem(System):
    """System description that adds jump operators for the simulation of dissipation, etc."""

    @abstractmethod
    def get_jump_operators(self) -> list[tuple[Array, Array]]:
        """
        Return a list tuples of decay rates and jump operators for each subsystem.

        Returns
        -------
        list[Tuple[Array, Array]]
            List of jump operators
        """
        pass
