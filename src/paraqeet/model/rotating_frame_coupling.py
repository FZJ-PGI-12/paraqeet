"""Coupling Hamiltonian in the rotating frame of drive."""

import jax.numpy as jnp
from jax import vmap

from paraqeet.model.coupling import TwoBodyCoupling
from paraqeet.model.hamiltonian import Hamiltonian
from paraqeet.quantity import Array, Quantity


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

    _subsystem_A: Hamiltonian
    _subsystem_B: Hamiltonian
    _coefficient: Quantity
    __diff_freq: Quantity

    def __init__(
        self,
        subsystem_A: Hamiltonian,
        subsystem_B: Hamiltonian,
        coefficient: Quantity,
        diffFreq: Quantity,
    ):
        super().__init__(subsystem_A, subsystem_B, coefficient, is_longitudinal=False)
        self.__diff_freq = diffFreq

    def get_parameters(self) -> list[Quantity]:
        """Return the coupling coeffecient and the difference frequency.

        NOTE - Optimization using relational quantities can be optimize the
        drive frequencies for the two subsystems.

        Parameters
        ----------
        list[Quantity]
            Returns the list of parameters of the system.

        """
        return [self._coefficient, self.__diff_freq]

    def __coupling_operators(self) -> list[Array]:
        """Return the annhilation operator. Special implementation for two subsystems."""
        if len(self.subsystems) > 2:
            raise NotImplementedError("No implementation for more than 2 subsystems.")
        dim = self.subsystem_A.dimension()
        annihilation_ops: list[Array] = [jnp.sqrt(jnp.diag(jnp.arange(1, dim), k=1))]
        dim = self.subsystem_B.dimension()
        annihilation_ops.append(jnp.sqrt(jnp.diag(jnp.arange(1, dim), k=1)).conj().T)
        return annihilation_ops

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
            shape as the subsystem's Hamiltonian.get_matrix_one_time: (n,n)
            with n the subsystem dimension.
        """
        annihilation_ops = self.__coupling_operators()

        annihilation_ops[0] *= self._coefficient.get_value() * jnp.exp(1j * self.__diff_freq.get_value() * t)
        annihilation_ops_conj = [a.conj().T for a in annihilation_ops]
        return [annihilation_ops, annihilation_ops_conj]

    def get_RWA_gradients(self, times: Array):
        """Compute the gradients of the coupling expression in the rotating frame, i.e.
        include a phase factor for several timesteps.
        """
        return vmap(self.get_RWA_gradients_one_time)(times)

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
        annihilation_ops = self.__coupling_operators()
        if self._is_optimized(self._coefficient):
            annihilation_ops[0] *= jnp.exp(1j * self.__diff_freq.get_value() * t)
            annihilationOps_conj = [a.conj().T for a in annihilation_ops]
            grads = [[annihilation_ops, annihilationOps_conj]]
        elif self._is_optimized(self.__diff_freq):
            annihilation_ops[0] *= self._coefficient.get_value() * 1j * t
            annihilationOps_conj = [a.conj().T for a in annihilation_ops]
            grads = [[annihilation_ops, annihilationOps_conj]]
        else:
            grads = [[[jnp.zeros_like(ann_op) for ann_op in annihilation_ops]] * 2]
        return grads
