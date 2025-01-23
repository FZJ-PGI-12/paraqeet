"""Class definition of a qubit model."""

import jax.numpy as jnp

from cthree.Quantity import Quantity
from cthree.model.Drive import Drive
from cthree.model.Hamiltonian import Hamiltonian


class Qubit(Hamiltonian):
    """Hamiltonian of a single qubit frequency/2 * sigma_z.

    The implementation uses the convention of having the excited state
    of the qubit as the first entry in the state. If you need a two-level
    system that is compatible with the projection of a higher-dimensional
    system (ground state as first entry), use a resonator and restrict its
    dimension to 2.

    Parameters
    ----------
    frequency : cthree.Quantity
        Frequency for characterizing the qubit.
    drives : List[cthree.model.Drive], optional
        List of time-dependent drives.

    """

    __frequency: Quantity
    __annihilationOp: jnp.ndarray
    __drift: jnp.array

    def __init__(self, frequency: Quantity, drives: list[Drive] = None):
        super().__init__(drives)
        self.__frequency = frequency
        self.__annihilationOp = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        )
        self.__drift = 0.5 * jnp.diag(jnp.array([1.0, -1.0]))

    def getFrequency(self) -> Quantity:
        """Get the frequency of the qubit."""
        return self.__frequency

    def setFrequency(self, frequency: Quantity) -> None:
        """Set the frequency of the qubit."""
        self.__frequency = frequency

    def get_parameters(self) -> list[Quantity]:
        """Get parameters of the model.

        Returns
        -------
        List[cthree.Quantity]
            Returns the list of parameters of the system.

        """
        return self._getDriveParameters() + [self.__frequency]

    def dimension(self) -> int:
        """Dimension of the qubit.

        Returns
        -------
        int
            Returns 2 as the dimension.

        """
        return 2

    def getMatrixOneTime(self, t: float) -> jnp.ndarray:
        """Get the drive matrix.

        Parameters
        ----------
        t : float
            One time stamp.

        Returns
        -------
        jax.numpy.ndarray
            The repeated drive matrix.

        """
        H = self.__frequency.get_value() * self.__drift
        return H + self._getDriveMatrixOneTime(self.__annihilationOp, t)

    def gradientOneTime(self, t: float) -> jnp.ndarray:
        """Get the gradient of the drive.

        Parameters
        ----------
        t : float
            One time stamp.

        Returns
        -------
        jax.numpy.ndarray
            Returns the gradients of the drive.

        """
        # Fetch the gradient of the drive
        derivatives = self._getDriveGradientsOneTime(self.__annihilationOp, t)

        # Combine with the derivative wrt the frequency
        if self._is_optimised(self.__frequency):
            H = self.__drift.reshape((1, 2, 2))
            derivatives = jnp.append(derivatives, H, axis=0)

        return derivatives
