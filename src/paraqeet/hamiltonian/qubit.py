"""Class definition of a qubit model."""

from typing import override

import jax.numpy as jnp

from paraqeet.exceptions import ConfigurationException
from paraqeet.hamiltonian.drive import Drive
from paraqeet.hamiltonian.hamiltonian import Hamiltonian
from paraqeet.hamiltonian.utils import sigma_minus, sigma_x, sigma_y, sigma_z
from paraqeet.quantity import Array, Quantity


class QubitHamiltonian(Hamiltonian):
    r"""Hamiltonian of a single qubit, -frequency / 2 * pauli_z + drive.

    The implementation uses the quantum information convention of having :math:`|0\rangle` = [1 0]^T
    as ground state and :math:`|1\rangle` = [0 1]^T as excited state, compatible with the projection
    of a higher-dimensional system. Hence, the Hamiltonian should be taken with a minus sign.
    """

    def __init__(
        self,
        frequency: Quantity,
        drives: list[Drive] | None = None,
    ) -> None:
        """
        Args:
            frequency: Frequency of the qubit.
            drives: List of time-dependent drives.
        """
        super().__init__(drives)
        self.frequency = frequency
        self._sigma_minus = sigma_minus()
        self._sigma_x = sigma_x()
        self._sigma_y = sigma_y()
        self._sigma_z = sigma_z()

    @override
    def dimension(self) -> int:
        """Return the dimension of the Hilbert space of the system.

        Returns:
            Hilbert space dimension.
        """
        return 2

    @property
    def sigma_minus(self) -> Array:
        """Return the sigma minus operator."""
        return self._sigma_minus

    @property
    def sigma_plus(self) -> Array:
        """Return the sigma plus operator."""
        return self._sigma_minus.T

    @property
    def sigma_x(self) -> Array:
        """Return the Pauli X operator."""
        return self._sigma_x

    @property
    def sigma_y(self) -> Array:
        """Return the Pauli Y operator."""
        return self._sigma_y

    @property
    def sigma_z(self) -> Array:
        """Return the Pauli Z operator."""
        return self._sigma_z

    @override
    def get_parameters(self) -> list[Quantity]:
        """Get parameters of the model.

        Returns:
            The list of parameters of the system.
        """
        return self.get_drive_parameters() + [self.frequency]

    @override
    def get_value(self, times: Array) -> Array:
        hamil_0 = (-self.frequency.get_value() * self._sigma_z / 2) * jnp.ones((*times.shape, 1, 1))
        hamil = hamil_0 + self.get_drive_matrix(times)
        return hamil

    @override
    def get_gradient(self, times: Array) -> Array:
        # Fetch the gradient of the drive
        derivatives = self.get_drive_gradients(times)

        # Combine with the derivative wrt the frequency
        if self._is_optimized(self.frequency):
            hamil = (-self._sigma_z / 2) * jnp.ones([*times.shape, 1, 1, 1])
            derivatives = jnp.append(derivatives, hamil, axis=1)
        return derivatives


class Qubit:
    """A system representing a qubit. It allows to store information about relaxation and dephasing times
    and get the corresponding jump operators.
    """

    def __init__(
        self,
        hamiltonian: QubitHamiltonian,
        t1: Quantity | None = None,
        temp: Quantity | None = None,
        t2star: Quantity | None = None,
    ) -> None:
        """
        Args:
            hamiltonian: The Hamiltonian of the qubit.
            t1: Energy relaxation time.
            temp: Temperature of the qubit.
            t2star: Dephasing time.
        """
        self.hamiltonian = hamiltonian
        self.t1 = t1
        self.temp = temp
        self.t2star = t2star

    def get_decay_rates(self) -> list[Array]:
        """Return decay rate for T1, T2star and Temp respectively."""
        if (self.t1 is None) or (self.t2star is None) or (self.temp is None):
            raise ConfigurationException("Specify values of T1, T2star and Temp for Open system simulations.")

        gamma = 1 / self.t1.get_value()
        gamma_t2star = 0.5 / self.t2star.get_value()

        hbar_over_kb = 7.638232582257738e-12
        beta = hbar_over_kb / (self.temp.get_value())
        nbar = jnp.exp(-beta * self.hamiltonian.frequency.get_value())
        gamma_temp = gamma * nbar
        gamma_t1 = gamma * (nbar + 1)
        return [gamma_t1, gamma_temp, gamma_t2star]

    def get_jump_operators(self) -> list[Array]:
        """Return a list of jump operators for the qubit."""
        sigma_minus = self.hamiltonian.sigma_minus
        sigma_plus = self.hamiltonian.sigma_plus
        gamma_t1, gamma_temp, gamma_t2star = self.get_decay_rates()
        col_t1 = jnp.sqrt(gamma_t1) * sigma_minus
        col_temp = jnp.sqrt(gamma_temp) * sigma_plus
        col_t2star = jnp.sqrt(gamma_t2star) * 2 * jnp.matmul(sigma_plus, sigma_minus)
        return [col_t1, col_temp, col_t2star]
