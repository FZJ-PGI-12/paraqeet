"""Class definition of the Drive Hamiltonian in the rotating frame of drive."""

from jax.numpy import ndarray
import jax.numpy as jnp

from cthree.Quantity import Quantity
from cthree.model.Drive import Drive
from cthree.signal.Generator import Generator


class RotatingFrameDrive(Drive):
    """Drive Hamiltonian in the Frame rotating at the frequency of the drive.

    __signalGenerator: Generator
        Signal Generator without a LO, like the PWCGenerator
    """

    __signalGenerator: Generator

    def __init__(self, signalGenerator: Generator):
        self.__signalGenerator = signalGenerator

    def getGenerator(self) -> Generator:
        """Get the signal generator from the system.

        Returns
        -------
        cthree.signal.Generator.Generator
            Returns the signal generator object from the system.

        """
        return self.__signalGenerator

    def getParameters(self) -> list[Quantity]:
        """Get a list of parameters of the system.

        Returns
        -------
        List[Quantity]
            List of optimizable parameters of the system.

        """
        return self.__signalGenerator.getParameters()

    def getMatrixOneTime(
        self, annihilationOperator: jnp.ndarray, t: float
    ) -> jnp.ndarray:
        r"""Implement drive in the rotating frame of drive.

        Drive Hamiltonian is implemented as
        \\big\\{ \\Omega a + \\Omega^* a^\\dagger \\big\\}
        Where \\Omega is the envelope (without the LO).

        TODO - Chcek this calculation

        Parameters
        ----------
        annihilationOperator: jnp.ndarray
            Annihilation operator of the subsystem
        t: float
            One time step
        """
        env = self.__signalGenerator.generateSignal(t)
        return (
            env * annihilationOperator
            + jnp.conjugate(env) * annihilationOperator.conj().T
        )

    def gradientOneTime(
        self, annihilationOperator: ndarray, t: float
    ) -> ndarray:
        """Get the one-time gradient of the system.

        Fetches the gradient from the drive and transforms it into the
        correct shape for the Hamiltonian.

        Parameters
        ----------
        a : numpy.ndarray
            Operator for longitudinal or transverse drive.
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            Returns the shape-shifted gradient from the drive.

        """
        envGrad = self.__signalGenerator.generateSignalGradient(t).reshape(
            (-1, 1, 1)
        )
        return (
            envGrad * annihilationOperator
            + jnp.conjugate(envGrad) * annihilationOperator.conj().T
        )
