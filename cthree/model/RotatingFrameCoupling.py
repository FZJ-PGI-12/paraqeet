"""Coupling Hamiltonian in the rotating frame of drive."""

from cthree.Quantity import Quantity
from cthree.model.Coupling import Coupling
from cthree.model.Hamiltonian import Hamiltonian
import numpy as np
import jax.numpy as jnp


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
    __diffFreq: Quantity

    def __init__(
        self,
        subsystems: list[Hamiltonian],
        coefficient: Quantity,
        diffFreq: Quantity,
    ):
        super().__init__(subsystems, coefficient, isLongitudinal=False)
        self.__diffFreq = diffFreq

    def get_parameters(self) -> list[Quantity]:
        """Return the coupling coeffecient and the difference frequency.

        NOTE - Optimisation using relational quantities can be optimise the
        drive frequencies for the two subsystems.

        Parameters
        ----------
        list[cthree.Quantity]
            Returns the list of parameters of the system.

        """
        return [self._coefficient, self.__diffFreq]

    def __couplingOperators(self) -> list[np.ndarray]:
        """Return the annhilation operator."""
        dimensions = [s.dimension() for s in self.getSubsystems()]
        annihilationOp = [np.sqrt(np.diag(np.arange(1, dim, dtype=np.float64), k=1)) for dim in dimensions]
        if len(annihilationOp) > 1:
            annihilationOp[1] = annihilationOp[1].conj().T
        return annihilationOp

    def getMatricesOneTime(self, t: float) -> list[np.ndarray]:
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
        List[List[jax.numpy.ndarray]]
            The outer list are the coupling terms. The inner list contains
            matrices for each subsystem. The matrices (ndarray) have the same
            shape as the subsystem's Hamiltonian.getMatrixOneTime: (n,n)
            with n the subsystem dimension.
        """
        annihilationOp = self.__couplingOperators()
        if len(annihilationOp) > 2:
            raise NotImplementedError()

        annihilationOp[0] *= self._coefficient.get_value() * jnp.exp(1j * self.__diffFreq.get_value() * t)
        annihilationOp_conj = [a.conj().T for a in annihilationOp]
        return [annihilationOp, annihilationOp_conj]

    def gradientOneTime(self, t) -> list[list[np.ndarray]]:
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
        List[List[List[jax.numpy.ndarray]]]
            The outer list represents the gradients with respect to
            all optimised parameters. The rest is in the same shape as the
            result of getMatricesOneTime.

        """
        if self._is_optimised(self._coefficient):
            annihilationOp = self.__couplingOperators()
            annihilationOp[0] *= jnp.exp(1j * self.__diffFreq.get_value() * t)
            annihilationOp_conj = [a.conj().T for a in annihilationOp]
            grads = [annihilationOp, annihilationOp_conj]
        elif self._is_optimised(self.__diffFreq):
            annihilationOp = self.__couplingOperators()
            annihilationOp[0] *= self._coefficient.get_value() * 1j * t
            annihilationOp_conj = [a.conj().T for a in annihilationOp]
            grads = [annihilationOp, annihilationOp_conj]
        else:
            grads = jnp.empty((0, self._totalDims, self._totalDims))
        return grads
