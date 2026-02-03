"""Class definition of the Resonator Hamiltonian model."""

import jax
import jax.numpy as jnp

from paraqeet.exceptions import ConfigurationException
from paraqeet.model.differentiable_hamiltonian import DifferentiableHamiltonian
from paraqeet.model.drive import Drive
from paraqeet.quantity import Array, Quantity

jax.config.update("jax_enable_x64", True)


class Resonator(DifferentiableHamiltonian):
    """Hamiltonian of a harmonic oscillator.

    The only optimizable parameter is the frequency.

    Parameters
    ----------
    dimension : int
        Dimension of the harmonic oscillator.
    frequency : Quantity
        Frequency of the harmonic oscillator.
    drives : list[Drive], optional
        List of time-dependent drives of the subsystem.

    """

    _dimension: int
    _frequency: Quantity
    _annihilation_op: Array
    _num_op: Array
    _t1: Quantity | None
    _temp: Quantity | None
    _t2star: Quantity | None

    def __init__(
        self,
        dimension: int,
        frequency: Quantity,
        drives: list[Drive] | None = None,
        t1: Quantity | None = None,
        temp: Quantity | None = None,
        t2star: Quantity | None = None,
    ):
        super().__init__(drives=drives)
        self._dimension = dimension
        self._frequency = frequency
        self._annihilation_op = jnp.sqrt(jnp.diag(jnp.arange(1, dimension, dtype=jnp.float64), k=1))
        self._num_op = self._annihilation_op.T @ self._annihilation_op
        self.t1 = t1
        self.temp = temp
        self.t2star = t2star

    def dimension(self):
        """Get the dimension of the resonator."""
        return self._dimension

    @property
    def frequency(self) -> Quantity:
        """Get the frequency of the resonator."""
        return self._frequency

    @frequency.setter
    def frequency(self, frequency: Quantity) -> None:
        """Set the frequency of the resonator."""
        self._frequency = frequency

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
        return self._get_drive_parameters() + [self._frequency]

    def get_value_at_timestep(self, timestep: float) -> Array:
        """Get the drive matrix.

        Parameters
        ----------
        timestep : float
            One time stamp.

        Returns
        -------
        Array
            The drive matrix at a single timestamp.

        """
        H = self._frequency.get_value() * self._num_op
        return H + self._get_drive_matrix_at_timestep(self._annihilation_op, timestep)

    def get_gradient_at_timestep(self, time: float) -> Array:
        """Get the gradient of the drive.

        Parameters
        ----------
        times : float
            One time stamp.

        Returns
        -------
        Array
            Returns the gradients of the drive.

        """
        # Fetch the gradient of the drive
        derivatives = self._get_drive_gradients_at_timestep(self._annihilation_op, time)

        # Combine with the derivative wrt the frequency
        if self._is_optimized(self._frequency):
            grad = self._num_op.reshape((1,) + self._num_op.shape)
            derivatives = jnp.append(derivatives, grad, axis=0)
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

    def get_collapseops(self) -> list[tuple[Array, Array]]:
        """
        Return a list tuples of decay rates and collapse operators for each subsystem.

        Return
        ------
        list[tuple[Array, Array]]
            List of collapse operators
        """
        gamma_t1, gamma_temp, gamma_t2star = self.get_decay_rates()
        col_t1 = self._annihilation_op
        col_temp = self._annihilation_op.T
        col_t2star = 2 * self._num_op
        return [(gamma_t1, col_t1), (gamma_temp, col_temp), (gamma_t2star, col_t2star)]
