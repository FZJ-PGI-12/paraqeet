"""Class definition of the Resonator Hamiltonian model."""

import jax.numpy as jnp
from paraqeet.quantity import Array

from paraqeet.model.drive import Drive
from paraqeet.model.hamiltonian import Hamiltonian
from paraqeet.quantity import Quantity

import jax

jax.config.update("jax_enable_x64", True)


class Resonator(Hamiltonian):
    """Hamiltonian of a harmonic oscillator.

    The only optimisable parameter is the frequency.

    Parameters
    ----------
    dimension : int
        Dimension of the harmonic oscillator.
    frequency : Quantity
        Frequency of the harmonic oscillator.
    drives : List[Drive], optional
        List of time-dependent drives of the subsystem.

    """

    __dimension: int
    __frequency: Quantity
    __annihilation_op: Array
    __numOp: Array
    __t1: Quantity | None
    __temp: Quantity | None
    __t2star: Quantity | None

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
        self.__dimension = dimension
        self.__frequency = frequency
        self.__annihilation_op = jnp.sqrt(jnp.diag(jnp.arange(1, dimension, dtype=jnp.float64), k=1))
        self.__numOp = self.__annihilation_op.T @ self.__annihilation_op
        self.t1 = t1
        self.temp = temp
        self.t2star = t2star

    def dimension(self):
        """Get the dimension of the resonator."""
        return self.__dimension

    @property
    def frequency(self) -> Quantity:
        """Get the frequency of the resonator."""
        return self.__frequency

    @frequency.setter
    def frequency(self, frequency: Quantity) -> None:
        """Set the frequency of the resonator."""
        self.__frequency = frequency

    @property
    def t1(self) -> Quantity | None:
        """Get the t1 of the resonator."""
        return self.__t1

    @t1.setter
    def t1(self, t1: Quantity | None) -> None:
        """Set the t1 of the resonator."""
        self.__t1 = t1

    @property
    def temp(self) -> Quantity | None:
        """Get the temp of the resonator."""
        return self.__temp

    @temp.setter
    def temp(self, temp: Quantity | None) -> None:
        """Set the temp of the resonator."""
        self.__temp = temp

    @property
    def t2star(self) -> Quantity | None:
        """Get the t2star of the resonator."""
        return self.__t2star

    @t2star.setter
    def t2star(self, t2star: Quantity | None) -> None:
        """Set the t2star of the resonator."""
        self.__t2star = t2star

    def get_parameters(self) -> list[Quantity]:
        """Get parameters of the model.

        Returns
        -------
        List[Quantity]
            Returns the list of parameters of the system.

        """
        return self._get_drive_parameters() + [self.__frequency]

    def get_matrix_one_time(self, t: Array) -> Array:
        """Get the drive matrix.

        Parameters
        ----------
        t : float
            One time stamp.

        Returns
        -------
        chtree.quantity.Array
            The drive matrix at a single timestamp.

        """
        H = self.__frequency.get_value() * self.__numOp
        return H + self._get_drive_matrix_one_time(self.__annihilation_op, t)

    def gradient_one_time(self, t: Array) -> Array:
        """Get the gradient of the drive.

        Parameters
        ----------
        t : float
            One time stamp.

        Returns
        -------
        chtree.quantity.Array
            Returns the gradients of the drive.

        """
        # Fetch the gradient of the drive
        derivatives = self._get_drive_gradients_one_time(self.__annihilation_op, t)

        # Combine with the derivative wrt the frequency
        if self._is_optimised(self.__frequency):
            grad = self.__numOp.reshape((1,) + self.__numOp.shape)
            derivatives = jnp.append(derivatives, grad, axis=0)

        return derivatives

    def get_decay_rates(self) -> list[Array]:
        """Return decay rate for T1, T2star and Temp respectively."""
        if (self.t1 is None) or (self.t2star is None) or (self.temp is None):
            raise Exception("Specify values of T1, T2star and Temp for Open system simulations.")

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
        List[Tuple[float, Array]]
            List of collapse operators
        """
        gamma_t1, gamma_temp, gamma_t2star = self.get_decay_rates()
        col_t1 = self.__annihilation_op
        col_temp = self.__annihilation_op.T
        col_t2star = 2 * self.__numOp
        return [(gamma_t1, col_t1), (gamma_temp, col_temp), (gamma_t2star, col_t2star)]
