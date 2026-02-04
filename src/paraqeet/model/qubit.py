"""Class definition of a qubit model."""

import jax.numpy as jnp

from paraqeet.exceptions import ConfigurationException
from paraqeet.model.differentiable_hamiltonian import DifferentiableHamiltonian
from paraqeet.model.drive import Drive
from paraqeet.quantity import Array, Quantity


# TODO: Why does Qubit inherit from Hamiltonian? Is a qubit really an operator? 
# What does a derivative of a qubit mean physically? Does get_value_at_timestep
# mean a measurement of the qubit at a certain time? 
class Qubit(DifferentiableHamiltonian):
    """Hamiltonian of a single qubit frequency/2 * sigma_z.

    The implementation uses the convention of having the excited state
    of the qubit as the first entry in the state. If you need a two-level
    system that is compatible with the projection of a higher-dimensional
    system (ground state as first entry), use a resonator and restrict its
    dimension to 2.

    Parameters
    ----------
    frequency : Quantity
        Frequency for characterizing the qubit.
    drives : list[Drive] | None
        List of time-dependent drives.

    """

    _frequency: Quantity
    _annihilation_op: Array
    _drift: Array
    _t1: Quantity | None
    _temp: Quantity | None
    _t2star: Quantity | None

    # TODO: should we move DifferentiableHamiltonian-object to the constructor instead of inheriting from it?
    def __init__(
        self,
        frequency: Quantity,
        drives: list[Drive] | None = None,
        t1: Quantity | None = None,
        temp: Quantity | None = None,
        t2star: Quantity | None = None,
    ):
        super().__init__(drives)
        self._frequency = frequency
        self._annihilation_op = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        )
        self._drift = 0.5 * jnp.diag(jnp.array([1.0, -1.0]))
        self.t1 = t1
        self.temp = temp
        self.t2star = t2star

    @property
    def frequency(self) -> Quantity:
        """Get the frequency of the qubit."""
        return self._frequency

    @frequency.setter
    def frequency(self, frequency: Quantity) -> None:
        """Set the frequency of the qubit."""
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
        list[Quantity]
            Returns the list of parameters of the system.

        """
        return self._get_drive_parameters() + [self._frequency]

    def dimension(self) -> int:
        """Dimension of the qubit.

        Returns
        -------
        int
            Returns 2 as the dimension.

        """
        return 2

    def get_value_at_timestep(self, timestep: float) -> Array:
        """Get the drive matrix.

        Parameters
        ----------
        timestep: Array
            One time stamp.

        Returns
        -------
        Array
            The repeated drive matrix.

        """
        hamil = self._frequency.get_value() * self._drift
        return hamil + self._get_drive_matrix_at_timestep(self._annihilation_op, timestep)

    def get_gradient_at_timestep(self, time: float) -> Array:
        """Get the matrix representations of value and gradient of the drive as a tuple.

        Parameters
        ----------
        times: Array
            Array of timestamps of interest.

        Returns
        -------
        tuple[Array, Array]
            Returns the value und gradients of the drive.

        """
        # Fetch the gradient of the drive
        derivatives = self._get_drive_gradients_at_timestep(self._annihilation_op, time)

        # Combine with the derivative wrt the frequency
        if self._is_optimized(self._frequency):
            hamil = self._drift.reshape((1, 2, 2))
            derivatives = jnp.append(derivatives, hamil, axis=0)
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
        """Return a list tuples of decay rates and collapse operators for each subsystem."""
        gamma_t1, gamma_temp, gamma_t2star = self.get_decay_rates()
        col_t1 = self._annihilation_op
        col_temp = self._annihilation_op.T
        col_t2star = 2 * jnp.matmul(self._annihilation_op.T, self._annihilation_op)
        return [(gamma_t1, col_t1), (gamma_temp, col_temp), (gamma_t2star, col_t2star)]
