"""Class definition of a coupling optimisable model."""

import jax
import jax.numpy as jnp
from paraqeet.quantity import Array
from jax import vmap

from paraqeet.model.hamiltonian import Hamiltonian
from paraqeet.optimisable import Optimisable
from paraqeet.quantity import Quantity

jax.config.update("jax_enable_x64", True)


class Coupling(Optimisable):
    """Create a coupling optimisable model.

    Represents the coupling of two or more subsystems in a composite
    Hamiltonian. This class implements longitudinal and transversal
    coupling with a constant scalar coefficient.
    The coefficient is the only optimisable parameter.
    Subclasses can alter the behavior by overriding the getMatrix function.

    Parameters
    ----------
    subsystems : List[Hamiltonian]
        The coupled subsystems.
    coefficient : Quantity
        Either a constant coefficient as float or a callable that returns the
        coefficient for a given time.
    is_longitudinal : bool
        Whether the coupling is longitudinal or transversal.
    useRWA : bool, optional
        If the transversal coupling should use the rotating-wave approximation
        or should include double excitation terms.
    """

    _subsystems: list[Hamiltonian]
    _coefficient: Quantity
    _total_dims: int
    __is_longitudinal: bool
    __useRWA: bool

    def __init__(
        self,
        subsystems: list[Hamiltonian],
        coefficient: Quantity,
        is_longitudinal: bool,
        useRWA: bool = False,
    ):
        self._subsystems = subsystems
        self._coefficient = coefficient
        self.__is_longitudinal = is_longitudinal
        self.__useRWA = useRWA
        self._total_dims = int(jnp.prod(jnp.array([s.dimension() for s in self.subsystems])))

    def get_parameters(self) -> list[Quantity]:
        """Collect parameters from all subsystems and couplings.

        Parameters
        ----------
        list[Quantity]
            Returns the list of parameters of the system.

        """
        return [self._coefficient]

    @property
    def subsystems(self) -> list[Hamiltonian]:
        """Return all subsystems that are coupled by this term.

        Returns
        -------
        List[Hamiltonian]
            List of subystems.

        """
        return self._subsystems

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
        matrices = self.__coupling_operators()
        for i in range(len(matrices)):  # iterating over terms
            matrices[i][0] *= self._coefficient.get_value()
        return matrices

    def get_matrices(self, t: Array) -> list[list[Array]]:
        """Return the matrices for an array of time.

        vmaps over the method for one time step.

        Parameters
        ----------
        t : chtree.quantity.Array
            Array of times

        Returns
        -------
        List[List[chtree.quantity.Array]]
            The outer list are the coupling terms. The inner list represents
            the subsystems. The matrices (Array) have the same shape as the
            subsystem's Hamiltonian.getMatrix: (t,n,n) with t the time and n
            the subsystem dimension.

        """
        # Technically, vmap returns "any" but we know the type of get_matrices_one_time is correct.
        return vmap(self.get_matrices_one_time)(t)  # type:ignore

    def gradient_one_time(self, t: Array) -> list[list[list[Array]]]:
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
        coup_ops = self.__coupling_operators()
        if self._is_optimised(self._coefficient):
            grads = [coup_ops]
        else:
            grads = [[[jnp.empty((0, 0)) for _ in sub] for sub in coup_ops]]
        return grads

    def gradient(self, t: Array) -> list[list[list[Array]]]:
        """Return the gradients for an array of times.

        Parameters
        ----------
        t : chtree.quantity.Array
            One-dimensional vector of timestamps.

        Returns
        -------
        List[List[chtree.quantity.Array]]
            The outer list represents the gradients with respect to all
            optimised parameters.
            The rest is in the same shape as the result of getMatrices.

        """
        # Technically, vmap returns "any" but we know the type of gradient_one_time is correct.
        return vmap(self.gradient_one_time)(t)  # type: ignore

    def __coupling_operators(self) -> list[list[Array]]:
        """Return coupling operators.

        Returns the operators of the longitudinal or transversal coupling
        without coefficients. A list of terms is returned which have to be
        summed over to produce the coupling Hamiltonian.
        In case of RWA, right now only 2 subsytems are supported.

        Returns
        -------
        List[List[Array]]
            A list of operators for each subsystem.
            The subsystems are the outer list.

        """
        if self.__is_longitudinal:
            # Number operator (a^\dagger a) for each subsystem
            return [[jnp.diag(jnp.arange(0, s.dimension(), dtype=jnp.float64)) for s in self._subsystems]]

        elif self.__useRWA:
            # TODO - How to use RWA for more than 2 subsystems?

            if len(self.subsystems) > 2:
                raise NotImplementedError("RWA is defined for 2 subsystems only")

            dimensions = [s.dimension() for s in self.subsystems]
            annihilationOp = [jnp.sqrt(jnp.diag(jnp.arange(1, dim, dtype=jnp.float64), k=1)) for dim in dimensions]
            return [
                [annihilationOp[0], annihilationOp[1].T],
                [annihilationOp[0].T, annihilationOp[1]],
            ]

        else:
            # (a + a^\dagger) for each subsystem
            dimensions = [s.dimension() for s in self.subsystems]
            annihilationOp = [jnp.sqrt(jnp.diag(jnp.arange(1, dim, dtype=jnp.float64), k=1)) for dim in dimensions]
            return [[(a + a.T) for a in annihilationOp]]
