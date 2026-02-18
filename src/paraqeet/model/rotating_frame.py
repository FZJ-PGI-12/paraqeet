"""Coupling Hamiltonian in the rotating frame of drive."""

import jax.numpy as jnp
from jax import vmap

from paraqeet.model.coupling import TwoBodyCoupling
from paraqeet.model.drive import Drive
from paraqeet.model.system import System
from paraqeet.quantity import Array, Quantity
from paraqeet.signal.generator import Generator


class RotatingFrameCoupling(TwoBodyCoupling):
    """Implements the coupling in the rotating frame of the drive.

    If multiple subsystems are coupled specify the difference frequency of the
    individual drive frames.

    NOTE - Right now this only works for TWO SUBSYSTEMS.
    TODO - Generalize this for multiple subsystems

    Parameters
    ----------
    subsystems: list[Hamiltonian]
        A set of Hamiltonians which represent the coupling
    coefficient: Quantity
        Constant drive coefficient.
    diffFreq: Quantity
        Diffrence of drive frequencies for multiple subsystems.
    """

    _subsystem_A: System
    _subsystem_B: System
    _coefficient: Quantity
    _diff_freq: Quantity

    def __init__(
        self,
        subsystem_A: System,
        subsystem_B: System,
        coefficient: Quantity,
        diffFreq: Quantity,
    ):
        super().__init__(subsystem_A, subsystem_B, coefficient, is_longitudinal=False)
        self._diff_freq = diffFreq

    def get_parameters(self) -> list[Quantity]:
        """Return the coupling coeffecient and the difference frequency.

        NOTE - Optimization using relational quantities can be optimize the
        drive frequencies for the two subsystems.

        Parameters
        ----------
        list[Quantity]
            Returns the list of parameters of the system.

        """
        return [self._coefficient, self._diff_freq]

    def _coupling_operators(self) -> list[list[Array]]:
        """Return the annhilation operator. Special implementation for two subsystems."""
        if len(self.subsystems) > 2:
            raise NotImplementedError("No implementation for more than 2 subsystems.")
        dim = self.subsystem_A.dimension()
        annihilation_ops: list[Array] = [jnp.sqrt(jnp.diag(jnp.arange(1, dim), k=1))]
        dim = self.subsystem_B.dimension()
        annihilation_ops.append(jnp.sqrt(jnp.diag(jnp.arange(1, dim), k=1)).conj().T)
        return annihilation_ops

    # TODO: is t a one time point or an array of time points?
    def get_RWA_couplings(self, t: Array) -> list[list[Array]]:
        """Return the matrix representation of the coupling for all subsystems.

        A list of terms in the coupling is returned, where each of the term
        contains operators for each subsystem. A composite Hamiltonian puts
        these operators in the correct position in the tensor space to create
        the operators and then sum over the terms.

        Parameters
        ----------
        t: Array
            One time step.

        Returns
        -------
        list[list[Array]]
            The outer list are the coupling terms. The inner list contains
            matrices for each subsystem. The matrices (Array) have the same
            shape as the subsystem's Hamiltonian.get_value_one_time: (n,n)
            with n the subsystem dimension.
        """
        annihilation_ops = self._coupling_operators()

        annihilation_ops[0] *= self._coefficient.get_value() * jnp.exp(1j * self._diff_freq.get_value() * t)
        annihilation_ops_conj = [a.conj().T for a in annihilation_ops]
        return [annihilation_ops, annihilation_ops_conj]

    def get_RWA_gradients(self, times: Array):
        """Compute the gradients of the coupling expression in the rotating frame, i.e.
        include a phase factor for several timesteps.
        """
        # ignoring mypy due to vmap
        return vmap(self.get_RWA_gradients_one_time)(times)  # type: ignore

    # TODO: should we rename this method to get_RWA_gradient_at_timestep for consistency?
    def get_RWA_gradients_one_time(self, t: float) -> list[list[list[Array]]]:
        """Get the one-time gradient of the matrix.

        Returns the gradient of the matrix representation of the coupling
        for all subsystems. Each entry in the list is the gradient with
        respect to one parameter, factorised into subsystems
        (representing a list of term in the coupling).

        Parameters
        ----------
        t: Array
            One time point.

        Returns
        -------
        list[list[list[Array]]]
            The outer list represents the gradients with respect to
            all optimized parameters. The rest is in the same shape as the
            result of get_matrices_one_time.

        """
        annihilation_ops = self._coupling_operators()
        if self._is_optimized(self._coefficient):
            annihilation_ops[0] *= jnp.exp(1j * self._diff_freq.get_value() * t)
            annihilationOps_conj = [a.conj().T for a in annihilation_ops]
            grads = [[annihilation_ops, annihilationOps_conj]]
        elif self._is_optimized(self._diff_freq):
            annihilation_ops[0] *= self._coefficient.get_value() * 1j * t
            annihilationOps_conj = [a.conj().T for a in annihilation_ops]
            grads = [[annihilation_ops, annihilationOps_conj]]
        else:
            grads = [[[jnp.zeros_like(ann_op) for ann_op in annihilation_ops]] * 2]
        return grads


class RotatingFrameDrive(Drive):
    """Drive Hamiltonian in the Frame rotating at the frequency of the drive.

    _signal_generator: Generator
        Signal Generator without a LO, like the PWCGenerator
    """

    _signal_generator: Generator

    def __init__(self, signal_generator: Generator):
        self._signal_generator = signal_generator

    @property
    def generator(self) -> Generator:
        """Get the signal generator from the system.

        Returns
        -------
        Generator
            Returns the signal generator object from the system.

        """
        return self._signal_generator

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
