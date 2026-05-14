"""Class definition of the Transmon Hamiltonian model."""

import jax
import jax.numpy as jnp

from paraqeet.exceptions import ConfigurationException
from paraqeet.model.drive import Drive
from paraqeet.model.system import OpenSystem
from paraqeet.quantity import Array, Quantity

jax.config.update("jax_enable_x64", True)


class Transmon(OpenSystem):
    """Hamiltonian of an anharmonic oscillator.

    Optimizable parameters are the ground frequency and the anharmonicity.

    Parameters
    ----------
    dimension: int
        Dimension of the anharmonic oscillator.
    frequency: Quantity
        Frequency of the anharmonic oscillator.
    anharmonicity: Quantity
        Anharmonicity of the oscillator.
    drives: list[Drive]
        List of time-dependent drives of the subsystem.
    t1: Quantity | None
        Energy relaxation time.
    temp: Quantity | None
        Temperature of the qubit.
    t2star: Quantity | None
        Dephasing time.
    """

    def __init__(
        self,
        num_levels: int,
        frequency: Quantity,
        anharmonicity: Quantity,
        drives: list[Drive] | None = None,
        t1: Quantity | None = None,
        temp: Quantity | None = None,
        t2star: Quantity | None = None,
    ):
        super().__init__(drives=drives)
        self._num_levels = num_levels
        self.frequency = frequency
        self.anharmonicity = anharmonicity
        self._annihilation_op = jnp.sqrt(jnp.diag(jnp.arange(1, num_levels, dtype=jnp.float64), k=1))
        self._num_op = self._annihilation_op.T @ self._annihilation_op
        self._anharmonic_term = 0.5 * self._num_op @ (self._num_op - jnp.eye(num_levels))
        self.t1 = t1
        self.temp = temp
        self.t2star = t2star

    def dimension(self) -> int:
        """Return the dimension of the Hilbert space of the system.

        Returns
        -------
        int
            Hilbert space dimension.

        """
        return self._num_levels

    @property
    def annihilation_op(self) -> Array:
        """Return the annihilation operator"""
        return self._annihilation_op

    @property
    def num_op(self) -> Array:
        """Return the Fock number operator"""
        return self._num_op

    @property
    def anharmonic_term(self) -> Array:
        """Return the anharmonic_term"""
        return self._anharmonic_term

    def get_parameters(self) -> list[Quantity]:
        """Get parameters of the model.

        Returns
        -------
        List[Quantity]
            Returns the list of parameters of the system.

        """
        return self.get_drive_parameters() + [
            self.frequency,
            self.anharmonicity,
        ]

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
        hamil_0 = self.frequency.get_value() * self._num_op + self.anharmonicity.get_value() * self._anharmonic_term
        hamil = hamil_0 + self.get_drive_matrix_at_timestep(t)
        return hamil

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
        # Fetch the gradient of the drive
        gradients = self.get_drive_gradients_at_timestep(t)

        # Combine with the derivatives wrt the frequency and anharmonicity
        grads_list = []
        if self._is_optimized(self.frequency):
            grads_list.append(self._num_op)
        if self._is_optimized(self.anharmonicity):
            grads_list.append(self._anharmonic_term)
        grads = jnp.stack(grads_list, axis=0) if len(grads_list) > 0 else jnp.empty((0,) + self._num_op.shape)
        gradients = jnp.append(gradients, grads, axis=0)
        return gradients

    def get_decay_rates(self) -> list[Array]:
        """Return decay rate for T1, T2star and Temp respectively."""
        if (self.t1 is None) or (self.t2star is None) or (self.temp is None):
            raise ConfigurationException("Specify values of T1, T2star and Temp for Open system simulations.")

        gamma = 1 / self.t1.get_value()
        gamma_t2star = 0.5 / self.t2star.get_value()

        hbar_over_kb = 7.638232582257738e-12
        beta = hbar_over_kb / (self.temp.get_value())

        freq = self.frequency.get_value()
        anharm = self.anharmonicity.get_value()
        if self.dimension() > 2:
            freq_diff = jnp.diag(jnp.array([freq + n * anharm for n in range(self.dimension())]), k=0)
            nbar = jnp.exp(-beta * freq_diff)
        else:
            nbar = jnp.exp(-beta * freq)
        gamma_temp = gamma * nbar
        gamma_t1 = gamma * (nbar + 1)
        return [gamma_t1, gamma_temp, gamma_t2star]

    def get_jump_operators(self) -> list[Array]:
        """
        Return a list of jump operators for the transmon.

        Return
        ------
        list[Array]
            List of jump operators
        """
        gamma_t1, gamma_temp, gamma_t2star = self.get_decay_rates()
        col_t1 = jnp.sqrt(gamma_t1) * self._annihilation_op
        col_temp = jnp.sqrt(gamma_temp) * self._annihilation_op.T
        col_t2star = jnp.sqrt(gamma_t2star) * 2 * jnp.matmul(self._annihilation_op.T, self._annihilation_op)
        return [col_t1, col_temp, col_t2star]
