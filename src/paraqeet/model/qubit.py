"""Class definition of a qubit model."""

from typing import override

import jax.numpy as jnp

from paraqeet.exceptions import ConfigurationException
from paraqeet.model.drive import Drive
from paraqeet.model.system import OpenSystem
from paraqeet.quantity import Array, Quantity


class Qubit(OpenSystem):
    r"""Hamiltonian of a single qubit -frequency / 2 * pauli_z.

    The implementation uses the quantum information convention of having :math:`|0\rangle` = [1 0]^T
    system that is compatible with the projection of a higher-dimensional
    as ground state and :math:`|1\rangle` = [0 1]^T as excited state. Hence, the Hamiltonian
    should be taken with a minus sign.

    Parameters
    ----------
    frequency: Quantity
        Frequency for characterizing the qubit.
    drives: list[Drive] | None
        List of time-dependent drives.
    t1: Quantity | None
        Energy relaxation time.
    temp: Quantity | None
        Temperature of the qubit.
    t2star: Quantity | None
        Dephasing time.
    """

    def __init__(
        self,
        frequency: Quantity,
        drives: list[Drive] | None = None,
        t1: Quantity | None = None,
        temp: Quantity | None = None,
        t2star: Quantity | None = None,
    ):
        super().__init__(drives)
        self.frequency = frequency
        self._sigma_minus = jnp.array(
            [
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )
        self._pauli_z = jnp.diag(jnp.array([1.0, -1.0]))
        self.t1 = t1
        self.temp = temp
        self.t2star = t2star

    @override
    def dimension(self) -> int:
        """Return the dimension of the Hilbert space of the system.

        Returns
        -------
        int
            Hilbert space dimension.

        """
        return 2

    @property
    def sigma_minus(self) -> Array:
        """Return the sigma minus operator"""
        return self._sigma_minus

    @property
    def sigma_plus(self) -> Array:
        """Return the sigme plus operator"""
        return self._sigma_minus.T

    @property
    def pauli_z(self) -> Array:
        """Return the Pauli Z operator"""
        return self._pauli_z

    @override
    def get_parameters(self) -> list[Quantity]:
        """Get parameters of the model.

        Returns
        -------
        list[Quantity]
            Returns the list of parameters of the system.

        """
        return self.get_drive_parameters() + [self.frequency]

    @override
    def get_value(self, times: Array) -> Array:
        hamil_0 = (-self.frequency.get_value() * self._pauli_z / 2) * jnp.ones((*times.shape, 1, 1))
        hamil = hamil_0 + self.get_drive_matrix(times)
        return hamil

    @override
    def get_gradient(self, times: Array) -> Array:
        # Fetch the gradient of the drive
        derivatives = self.get_drive_gradients(times)

        # Combine with the derivative wrt the frequency
        if self._is_optimized(self.frequency):
            hamil = (-self._pauli_z / 2) * jnp.ones([*times.shape, 1, 1, 1])
            derivatives = jnp.append(derivatives, hamil, axis=1)
        return derivatives

    def get_decay_rates(self) -> list[Array]:
        """Return decay rate for T1, T2star and Temp respectively."""
        if (self.t1 is None) or (self.t2star is None) or (self.temp is None):
            raise ConfigurationException("Specify values of T1, T2star and Temp for Open system simulations.")

        gamma = 1 / self.t1.get_value()
        gamma_t2star = 0.5 / self.t2star.get_value()

        hbar_over_kb = 7.638232582257738e-12
        beta = hbar_over_kb / (self.temp.get_value())
        nbar = jnp.exp(-beta * self.frequency.get_value())
        gamma_temp = gamma * nbar
        gamma_t1 = gamma * (nbar + 1)
        return [gamma_t1, gamma_temp, gamma_t2star]

    def get_jump_operators(self) -> list[Array]:
        """Return a list of jump operators for the qubit."""
        gamma_t1, gamma_temp, gamma_t2star = self.get_decay_rates()
        col_t1 = jnp.sqrt(gamma_t1) * self._sigma_minus
        col_temp = jnp.sqrt(gamma_temp) * self._sigma_minus.T
        col_t2star = jnp.sqrt(gamma_t2star) * 2 * jnp.matmul(self._sigma_minus.T, self._sigma_minus)
        return [col_t1, col_temp, col_t2star]
