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

    """

    _dimension: int
    _frequency: Quantity
    _anharmonicity: Quantity
    _annihilation_op: Array
    _num_op: Array
    _anharmonic_term: Array
    _t1: Quantity | None
    _temp: Quantity | None
    _t2star: Quantity | None

    def __init__(
        self,
        dimension: int,
        frequency: Quantity,
        anharmonicity: Quantity,
        drives: list[Drive] | None = None,
        t1: Quantity | None = None,
        temp: Quantity | None = None,
        t2star: Quantity | None = None,
    ):
        super().__init__(drives=drives)
        self._dimension = dimension
        self._frequency = frequency
        self._anharmonicity = anharmonicity
        self._annihilation_op = jnp.sqrt(jnp.diag(jnp.arange(1, dimension, dtype=jnp.float64), k=1))
        self._num_op = self._annihilation_op.T @ self._annihilation_op
        self._anharmonic_term = 0.5 * self._num_op @ (self._num_op - jnp.eye(self._dimension))
        self.t1 = t1
        self.temp = temp
        self.t2star = t2star

    def dimension(self) -> int:
        """Get the dimension of the Transmon system."""
        return self._dimension

    @property
    def frequency(self) -> Quantity:
        """Get the frequency of the Transmon system."""
        return self._frequency

    @frequency.setter
    def frequency(self, frequency: Quantity) -> None:
        """Set the frequency of the Transmon system."""
        self._frequency = frequency

    @property
    def anharmonicity(self) -> Quantity:
        """Get the anharmonicity of the Transmon system."""
        return self._anharmonicity

    @anharmonicity.setter
    def anharmonicity(self, anharmonicity: Quantity) -> None:
        """Set the anharmonicity of the Transmon system."""
        self._anharmonicity = anharmonicity

    @property
    def t1(self) -> Quantity | None:
        """Get the t1 of the resonator."""
        return self._t1

    @t1.setter
    def t1(self, t1: Quantity | None) -> None:
        """Set the t1 of the resonator."""
        self._t1 = t1

    @property
    def temp(self) -> Quantity | None:
        """Get the temp of the resonator."""
        return self._temp

    @temp.setter
    def temp(self, temp: Quantity | None) -> None:
        """Set the temp of the resonator."""
        self._temp = temp

    @property
    def t2star(self) -> Quantity | None:
        """Get the t2star of the resonator."""
        return self._t2star

    @t2star.setter
    def t2star(self, t2star: Quantity | None) -> None:
        """Set the t2star of the resonator."""
        self._t2star = t2star

    def get_parameters(self) -> list[Quantity]:
        """Get parameters of the model.

        Returns
        -------
        List[Quantity]
            Returns the list of parameters of the system.

        """
        return self._get_drive_parameters() + [
            self._frequency,
            self._anharmonicity,
        ]

    def get_hamiltonian_at_timestep(self, timestep: float) -> Array:
        """Get the drive matrix.

        Parameters
        ----------
        timestep : Array
            Vector of time samples.

        Returns
        -------
        Array
            The repeated drive matrix.

        """
        hamil = self._frequency.get_value() * self._num_op + self._anharmonicity.get_value() * self._anharmonic_term
        return hamil + self._get_drive_matrix_at_timestep(self._annihilation_op, timestep)

    def get_hamiltonian_gradient_at_timestep(self, time: float) -> Array:
        """Get the gradient of the drive.

        Parameters
        ----------
        t: Array
            Single time stamp.

        Returns
        -------
        Array
            Returns the gradients of the drive.

        """
        # Fetch the gradient of the drive
        gradients = self._get_drive_gradients_at_timestep(self._annihilation_op, time)

        # Combine with the derivatives wrt the frequency and anharmonicity
        grads_list = []
        if self._is_optimized(self._frequency):
            grads_list.append(self._num_op)
        if self._is_optimized(self._anharmonicity):
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

    def get_jump_operators(self) -> list[tuple[Array, Array]]:
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
