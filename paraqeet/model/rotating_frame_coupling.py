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
    subsystems : Set[Hamiltonian]
        A set of Hamiltonians which represent the coupling
    coefficient : Quantity
        Constant drive coefficient.
    diffFreq : Quantity
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
        annihilationOps: list[Array] = [jnp.sqrt(jnp.diag(jnp.arange(1, dim), k=1))]
        if len(self.subsystems) == 2:
            dim = self.subsystems[1].dimension()
            annihilationOps.append(jnp.sqrt(jnp.diag(jnp.arange(1, dim), k=1)).conj().T)
        return annihilationOps

    def get_matrices_one_time(self, t: Array) -> list[list[Array]]:
        """Return the matrix representation of the coupling for all subsystems.

        A list of terms in the coupling is returned, where each of the term
        contains operators for each subsystem. A composite Hamiltonian puts
        these operators in the correct position in the tensor space to create
        the operators and then sum over the terms.

        Parameters
        ----------
        t : float
            One time step.

        Returns
        -------
        List[List[chtree.quantity.Array]]
            The outer list are the coupling terms. The inner list contains
            matrices for each subsystem. The matrices (Array) have the same
            shape as the subsystem's Hamiltonian.getMatrixOneTime: (n,n)
            with n the subsystem dimension.
        """
        annihilationOps = self.__coupling_operators()
        if len(annihilationOps) > 2:
            raise NotImplementedError()

        annihilationOps[0] *= self._coefficient.get_value() * jnp.exp(1j * self.__diff_freq.get_value() * t)
        annihilationOps_conj = [a.conj().T for a in annihilationOps]
        return [annihilationOps, annihilationOps_conj]

    def gradient_one_time(self, t) -> list[list[list[Array]]]:
        """Get the one-time gradient of the matrix.

        Returns the gradient of the matrix representation of the coupling
        for all subsystems. Each entry in the list is the gradient with
        respect to one parameter, factorised into subsystems
        (representing a list of term in the coupling).

        Parameters
        ----------
        t : float
            One time point.

        Returns
        -------
        List[List[List[chtree.quantity.Array]]]
            The outer list represents the gradients with respect to
            all optimised parameters. The rest is in the same shape as the
            result of getMatricesOneTime.

        """
        if self._is_optimised(self._coefficient):
            annihilationOps = self.__coupling_operators()
            annihilationOps[0] *= jnp.exp(1j * self.__diff_freq.get_value() * t)
            annihilationOps_conj = [a.conj().T for a in annihilationOps]
            grads = [[annihilationOps, annihilationOps_conj]]
        elif self._is_optimised(self.__diff_freq):
            annihilationOps = self.__coupling_operators()
            annihilationOps[0] *= self._coefficient.get_value() * 1j * t
            annihilationOps_conj = [a.conj().T for a in annihilationOps]
            grads = [[annihilationOps, annihilationOps_conj]]
        else:
            grads = [[[jnp.zeros_like(annOp) for annOp in annihilationOps]] * 2]
        return grads
