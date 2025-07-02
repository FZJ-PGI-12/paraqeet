"""Class definition of the Drive Hamiltonian in the rotating frame of drive."""

import jax.numpy as jnp

from paraqeet.model.drive import Drive
from paraqeet.quantity import Quantity, Array
from paraqeet.signal.generator import Generator


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
        paraqeet.signal.generator.Generator
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
        return self.generator.get_parameters()

    def get_matrix_one_time(self, annihilation_operator: Array, t: Array) -> Array:
        r"""Implement drive in the rotating frame of drive.

        Drive Hamiltonian is implemented as
        \\big\\{ \\Omega a + \\Omega^* a^\\dagger \\big\\}
        Where \\Omega is the envelope (without the LO).

        Parameters
        ----------
        annihilation_operator: Array
            Annihilation operator of the subsystem
        t: Array
            One time step
        """
        env = self.generator.generate_signal(jnp.array([t]))
        return env * annihilation_operator + jnp.conjugate(env) * annihilation_operator.conj().T

    def gradient_one_time(self, annihilation_operator: Array, t: Array) -> Array:
        """Get the one-time gradient of the system.

        Fetches the gradient from the drive and transforms it into the
        correct shape for the Hamiltonian.

        Parameters
        ----------
        annihilation_operator : Array
            Operator for longitudinal or transverse drive.
        t : Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the shape-shifted gradient from the drive.

        """
        envGrad = self.generator.generate_signal_gradient(jnp.array([t])).reshape((-1, 1, 1))
        return envGrad * annihilation_operator + jnp.conjugate(envGrad) * annihilation_operator.conj().T
