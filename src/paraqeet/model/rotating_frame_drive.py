"""Class definition of the Drive Hamiltonian in the rotating frame of drive."""

import jax.numpy as jnp

from paraqeet.model.drive import Drive
from paraqeet.quantity import Array, Quantity
from paraqeet.signal.generator import Generator


class RotatingFrameDrive(Drive):
    """Drive Hamiltonian in the Frame rotating at the frequency of the drive.

    __signal_generator: Generator
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
        Generator
            Returns the signal generator object from the system.

        """
        return self.__signal_generator

    def get_parameters(self) -> list[Quantity]:
        """Get a list of parameters of the system.

        Returns
        -------
        list[Quantity]
            List of optimizable parameters of the system.

        """
        return self.generator.get_parameters()

    def get_value_at_timestep(self, annihilation_operator: Array, t: float) -> Array:
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
        env = self.generator.get_value(jnp.array([t]))
        return env * annihilation_operator + jnp.conjugate(env) * annihilation_operator.conj().T

    def get_gradient_at_timestep(self, annihilation_operator: Array, timestep: float) -> Array:
        """Get the one-time gradient of the system.

        Fetches the gradient from the drive and transforms it into the
        correct shape for the Hamiltonian.

        Parameters
        ----------
        annihilation_operator: Array
            Operator for longitudinal or transverse drive.
        timestep: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the shape-shifted gradient from the drive.

        """
        _, env_grad = self.generator.get_value_and_gradient(jnp.array([timestep]))
        env_grad = env_grad.reshape((-1, 1, 1))
        return env_grad * annihilation_operator + jnp.conjugate(env_grad) * annihilation_operator.conj().T
