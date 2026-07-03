"""Class definition for a matrix representation of a Hamiltonian."""

from abc import abstractmethod
from typing import override

import jax.numpy as jnp

from paraqeet.differentiable import Differentiable
from paraqeet.model.drive import Drive
from paraqeet.optimizable import Optimizable
from paraqeet.quantity import Array, Quantity


class Hamiltonian(Optimizable, Differentiable):
    """Class definition for a matrix representation of a Hamiltonian.

    Implementations can contain subsystems, couplings, and drive lines
    and have to take care of frame transformations.

    Attributes
    ----------
    drives : list[Drive]
        List of time-dependent drives.

    """

    drives: list[Drive]

    def __init__(self, drives: list[Drive] | None = None):
        self.drives = [d for d in drives if d is not None] if drives else []

    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the Hilbert space of the system.

        Returns
        -------
        int
            Hilbert space dimension.

        """
        pass

    @override
    @abstractmethod
    def get_value(self, times: Array) -> Array:
        """Calculate the Hamiltonian at different times.

        Parameters
        ----------
        times: Array
            Array of times.

        Returns:
        ----------
            The value of the Hamiltonian matrix at different times. The dimension should be
            (n_times, dimension, dimension).
        """
        pass

    @override
    @abstractmethod
    def get_gradient(self, times: Array) -> Array:
        """Calculate the gradient of the Hamiltonian at different times.

        Parameters
        ----------
            times: Array of times.

        Returns:
        ----------
            The gradient of the Hamiltonian. The dimension should be
            (n_times, n_params, dimension, dimension).
        """
        pass

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

        This function can be used be Hamiltonian implementations
        for including the drive.

        Parameters
        ----------
        times: Array
            Array of times.

        Returns
        -------
        Array
            Returns the sum of all drives in matrix form.

        """
        dim = self.dimension()
        mat = jnp.zeros((dim, dim))
        for drive in self.drives:
            mat += drive.get_value(times)
        return mat

    def get_drive_gradients(self, times: Array) -> Array:
        """Return the gradients of all drives.

        This function can be used by Hamiltonian implementations
        for including the drive gradients.

        Parameters
        ----------
        times: Array
            Array of time samples.

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
