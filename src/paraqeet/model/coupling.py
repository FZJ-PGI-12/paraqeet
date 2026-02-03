"""Class definition of a coupling optimizable model."""

import jax
import jax.numpy as jnp

from paraqeet.model.differentiable_hamiltonian import DifferentiableHamiltonian
from paraqeet.optimizable import Optimizable
from paraqeet.quantity import Array, Quantity

jax.config.update("jax_enable_x64", True)


# TODO: Make this a 2-system coupling class, finish in notebooks
class TwoBodyCoupling(Optimizable):
    """Create a coupling optimizable model.

    Represents the coupling of two or more subsystems in a composite
    Hamiltonian. This class implements longitudinal and transversal
    coupling with a constant scalar coefficient.
    The coefficient is the only optimizable parameter.
    Subclasses can alter the behavior by overriding the getMatrix function.

    Parameters
    ----------
    subsystems : list[Hamiltonian]
        The coupled subsystems.
    coefficient : Quantity
        Either a constant coefficient as float or a callable that returns the
        coefficient for a given time.
    is_longitudinal : bool
        Whether the coupling is longitudinal or transversal.
    use_rwa : bool, optional
        If the transversal coupling should use the rotating-wave approximation
        or should include double excitation terms.
    """

    _subsystem_A: DifferentiableHamiltonian
    _subsystem_B: DifferentiableHamiltonian
    _coefficient: Quantity
    _total_dims: int
    _is_longitudinal: bool
    _use_rwa: bool

    def __init__(
        self,
        subsystem_A: DifferentiableHamiltonian,
        subsystem_B: DifferentiableHamiltonian,
        coefficient: Quantity,
        is_longitudinal: bool,
        use_rwa: bool = False,
    ):
        self._subsystem_A = subsystem_A
        self._subsystem_B = subsystem_B
        self._coefficient = coefficient
        self._is_longitudinal = is_longitudinal
        self._use_rwa = use_rwa
        self._total_dims = subsystem_A.dimension() * subsystem_B.dimension()
        self._dims = [subsystem_A.dimension(), subsystem_B.dimension()]
        self._annihilation_op = [jnp.sqrt(jnp.diag(jnp.arange(1, dim, dtype=jnp.float64), k=1)) for dim in self._dims]

    def get_parameters(self) -> list[Quantity]:
        """Collect parameters from all subsystems and couplings.

        Parameters
        ----------
        list[Quantity]
            Returns the list of parameters of the system.

        """
        return [self._coefficient]

    @property
    def subsystem_A(self) -> DifferentiableHamiltonian:
        """Return subsystem A that is coupled by this term.

        Returns
        -------
        list[Hamiltonian]
            List of subystems.

        """
        return self._subsystem_A

    @property
    def subsystem_B(self) -> DifferentiableHamiltonian:
        """Return subsystem A that is coupled by this term.

        Returns
        -------
        list[Hamiltonian]
            List of subystems.

        """
        return self._subsystem_B

    @property
    def subsystems(self) -> list[DifferentiableHamiltonian]:
        """Return the subsystems as a list to be compatible with other couplings with potentially more subsystems."""
        return [self.subsystem_A, self.subsystem_B]

    def get_couplings(self) -> list[list[Array]]:
        """Return the matrix representation of a time-indepedent coupling.

        Returns
        -------
        list[Array]
            The outer list are the coupling terms. The inner list represents
            the subsystems. The matrices (Array) have the same shape as the
            subsystem's Hamiltonian.get_value: (t,n,n) with t the time and n
            the subsystem dimension.

        """
        matrices = self._coupling_operators()
        for i in range(len(matrices)):  # iterating over terms
            matrices[i][0] *= self._coefficient.get_value()
        return matrices

    def get_coupling_gradients(self) -> list[list[list[Array]]]:
        """Return the gradients of the coupling terms.

        Returns
        -------
        list[Array]
            The outer list represents the gradients with respect to all
            optimized parameters.
            The rest is in the same shape as the result of get_matrices.

        """
        coup_ops = self._coupling_operators()
        if self._is_optimized(self._coefficient):
            grads = [coup_ops]
        else:
            grads = [[[jnp.empty((0, 0)) for _ in sub] for sub in coup_ops]]
        return grads

    def _coupling_operators(self) -> list[list[Array]]:
        """Return coupling operators.

        Returns the operators of the longitudinal or transversal coupling
        without coefficients. A list of terms is returned which have to be
        summed over to produce the coupling Hamiltonian.
        In case of RWA, right now only 2 subsytems are supported.

        Returns
        -------
        list[list[Array]]
            A list of operators for each subsystem.
            The subsystems are the outer list.

        """
        if self._is_longitudinal:
            # Number operator (a^\dagger a) for each subsystem
            return [[jnp.diag(jnp.arange(0, s, dtype=jnp.float64)) for s in self._dims]]

        elif self._use_rwa:
            annihilation_op = self._annihilation_op
            return [
                [annihilation_op[0], annihilation_op[1].T],
                [annihilation_op[0].T, annihilation_op[1]],
            ]
        else:
            # (a + a^\dagger) for each subsystem
            return [[(a + a.T) for a in self._annihilation_op]]
