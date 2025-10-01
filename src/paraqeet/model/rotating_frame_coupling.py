"""Coupling Hamiltonian in the rotating frame of drive."""

import jax.numpy as jnp

from paraqeet.model.coupling import Coupling
from paraqeet.model.hamiltonian import Hamiltonian
from paraqeet.quantity import Quantity, Array


class RotatingFrameCoupling(Coupling):
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

    _subsystems: list[Hamiltonian]
    _coefficient: Quantity
    __diff_freq: Quantity

    def __init__(
        self,
        subsystems: list[Hamiltonian],
        coefficient: Quantity,
        diffFreq: Quantity,
    ):
        super().__init__(subsystems, coefficient, is_longitudinal=False)
        self.__diff_freq = diffFreq

    def get_parameters(self) -> list[Quantity]:
        """Return the coupling coeffecient and the difference frequency.

        NOTE - Optimisation using relational quantities can be optimise the
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
        dim = self.subsystems[0].dimension()
        annihilation_ops: list[Array] = [jnp.sqrt(jnp.diag(jnp.arange(1, dim), k=1))]
        if len(self.subsystems) == 2:
            dim = self.subsystems[1].dimension()
            annihilation_ops.append(jnp.sqrt(jnp.diag(jnp.arange(1, dim), k=1)).conj().T)
        return annihilation_ops

    def get_matrices_one_time(self, t: Array) -> list[list[Array]]:
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
        if len(annihilation_ops) > 2:
            raise NotImplementedError()

        annihilation_ops[0] *= self._coefficient.get_value() * jnp.exp(1j * self.__diff_freq.get_value() * t)
        annihilation_ops_conj = [a.conj().T for a in annihilation_ops]
        return [annihilation_ops, annihilation_ops_conj]

    def gradient_one_time(self, t: Array) -> list[list[list[Array]]]:
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
            all optimised parameters. The rest is in the same shape as the
            result of get_matrices_one_time.

        """
        annihilation_ops = self.__coupling_operators()
        if self._is_optimised(self._coefficient):
            annihilation_ops[0] *= jnp.exp(1j * self.__diff_freq.get_value() * t)
            annihilationOps_conj = [a.conj().T for a in annihilation_ops]
            grads = [[annihilation_ops, annihilationOps_conj]]
        elif self._is_optimised(self.__diff_freq):
            annihilation_ops[0] *= self._coefficient.get_value() * 1j * t
            annihilationOps_conj = [a.conj().T for a in annihilation_ops]
            grads = [[annihilation_ops, annihilationOps_conj]]
        else:
            grads = [[[jnp.zeros_like(ann_op) for ann_op in annihilation_ops]] * 2]
        return grads
