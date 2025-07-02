"""Class definition of a qubit model."""

import jax.numpy as jnp
from paraqeet.quantity import Array

from paraqeet.model.drive import Drive
from paraqeet.model.hamiltonian import Hamiltonian
from paraqeet.quantity import Quantity


class Qubit(Hamiltonian):
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
    drives : List[Drive], optional
        List of time-dependent drives.

    """

    __frequency: Quantity
    __annihilationOp: Array
    __drift: Array
    __t1: Quantity | None
    __temp: Quantity | None
    __t2star: Quantity | None

    def __init__(
        self,
        frequency: Quantity,
        drives: list[Drive] | None = None,
        t1: Quantity | None = None,
        temp: Quantity | None = None,
        t2star: Quantity | None = None,
    ):
        super().__init__(drives)
        self.__frequency = frequency
        self.__annihilationOp = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        )
        self.__drift = 0.5 * jnp.diag(jnp.array([1.0, -1.0]))
        self.t1 = t1
        self.temp = temp
        self.t2star = t2star

    @property
    def frequency(self) -> Quantity:
        """Get the frequency of the qubit."""
        return self.__frequency

    @frequency.setter
    def frequency(self, frequency: Quantity) -> None:
        """Set the frequency of the qubit."""
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

    def dimension(self) -> int:
        """Dimension of the qubit.

        Returns
        -------
        int
            Returns 2 as the dimension.

        """
        return 2

    def get_matrix_one_time(self, t: Array) -> Array:
        """Get the drive matrix.

        Parameters
        ----------
        t : float
            One time stamp.

        Returns
        -------
        chtree.quantity.Array
            The repeated drive matrix.

        """
        H = self.__frequency.get_value() * self.__drift
        return H + self._get_drive_matrix_one_time(self.__annihilationOp, t)

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
        derivatives = self._get_drive_gradients_one_time(self.__annihilationOp, t)

        # Combine with the derivative wrt the frequency
        if self._is_optimised(self.__frequency):
            H = self.__drift.reshape((1, 2, 2))
            derivatives = jnp.append(derivatives, H, axis=0)

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
        """Return a list tuples of decay rates and collapse operators for each subsystem."""
        gamma_t1, gamma_temp, gamma_t2star = self.get_decay_rates()
        col_t1 = self.__annihilationOp
        col_temp = self.__annihilationOp.T
        col_t2star = 2 * jnp.matmul(self.__annihilationOp.T, self.__annihilationOp)
        return [(gamma_t1, col_t1), (gamma_temp, col_temp), (gamma_t2star, col_t2star)]
