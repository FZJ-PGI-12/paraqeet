"""Class definition of the Drive Hamiltonian in the rotating frame of drive."""

from jax.numpy import ndarray
import jax.numpy as jnp

from cthree.quantity import Quantity
from cthree.model.drive import Drive
from cthree.signal.generator import Generator


class RotatingFrameDrive(Drive):
    """Drive Hamiltonian in the Frame rotating at the frequency of the drive.

    __signalGenerator: Generator
        Signal Generator without a LO, like the PWCGenerator
    """

    __signal_generator: Generator

    def __init__(self, signal_generator: Generator):
        self.__signal_generator = signal_generator

    @property
    def generator(self) -> Generator:
        """Get the signal generator from the system.

        Returns
        -------
        cthree.signal.generator.Generator
            Returns the signal generator object from the system.

        """
        return self.__signal_generator

    def get_parameters(self) -> list[Quantity]:
        """Get a list of parameters of the system.

        Returns
        -------
        List[Quantity]
            List of optimizable parameters of the system.

        """
        return self.__signal_generator.get_parameters()

    def get_matrix_one_time(self, annihilation_operator: jnp.ndarray, t: float) -> jnp.ndarray:
        r"""Implement drive in the rotating frame of drive.

        Drive Hamiltonian is implemented as
        \\big\\{ \\Omega a + \\Omega^* a^\\dagger \\big\\}
        Where \\Omega is the envelope (without the LO).

        TODO - Chcek this calculation

        Parameters
        ----------
        annihilation_operator: jnp.ndarray
            Annihilation operator of the subsystem
        t: float
            One time step
        """
        env = self.__signal_generator.generate_signal(jnp.array([t]))
        return env * annihilation_operator + jnp.conjugate(env) * annihilation_operator.conj().T

    def gradient_one_time(self, annihilation_operator: ndarray, t: float) -> ndarray:
        """Get the one-time gradient of the system.

        Fetches the gradient from the drive and transforms it into the
        correct shape for the Hamiltonian.

        Parameters
        ----------
        annihilation_operator : numpy.ndarray
            Operator for longitudinal or transverse drive.
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            Returns the shape-shifted gradient from the drive.

        """
        envGrad = self.__signal_generator.generate_signal_gradient(jnp.array([t])).reshape((-1, 1, 1))
        return envGrad * annihilation_operator + jnp.conjugate(envGrad) * annihilation_operator.conj().T
