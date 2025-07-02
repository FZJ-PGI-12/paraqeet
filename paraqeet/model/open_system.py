"""Class definition of an open system."""

from collections.abc import Callable
from paraqeet.quantity import Array, Quantity
from paraqeet.model.equation_of_motion import EquationOfMotion
from paraqeet.model.hamiltonian import Hamiltonian

import jax.numpy as jnp
from jax import vmap, jit
from jax.experimental.sparse import BCOO


class OpenSystem(EquationOfMotion):
    """
    Model of an open quantum system, defined by the Hamiltonian and collapse operators.
    Its dynamics given by the Lindblad master equation.

    Currently the gradients for ODE propagation methods is not supported.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Matrix representation of a Hamiltonian.
    sparse_superop: bool
        Flag to save superoperator as sparse matrices.
    ode_propagation: bool
        Flag to use ODE methods for propgation.
        If `true` then `get_matrix` method returns list of Hamiltonian (with time) and collapse operator.
        Else returns Lindblad superoperator.
    """

    _ode_propagation: bool
    __sparse_superop: bool
    _get_matrix_method: Callable

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        sparse_superop: bool = False,
        ode_propagation: bool = False,
    ):
        super().__init__(hamiltonian)
        self.__sparse_superop = sparse_superop
        self.ode_propagation = ode_propagation

    @property
    def sparse_superop(self) -> bool:
        """Flag to store superoperators as sparse matrices.

        Returns
        -------
        bool
            Flag to store sparse matrices.
        """
        return self.__sparse_superop

    @sparse_superop.setter
    def sparse_superop(self, sparse_superop: bool) -> None:
        self.__sparse_superop = sparse_superop

    @property
    def ode_propagation(self) -> bool:
        """Flag to set method of propagation to ODE.

        Returns
        -------
        bool
            Flag to use ODE propagation.
        """
        return self._ode_propagation

    @ode_propagation.setter
    def ode_propagation(self, ode_propagation: bool) -> None:
        self._ode_propagation = ode_propagation

        if ode_propagation:
            self._get_matrix_method = self.__get_ode_propagation_eom
        else:
            self._get_matrix_method = vmap(self.__create_lindbladian_superop)

    def get_parameters(self) -> list[Quantity]:
        """Get a list of optimisable parameters.

        Returns
        -------
        list[Quantity]
            list of optimizable parameters of the system.

        """
        return self._hamiltonian.get_parameters()

    def get_collapseops(self) -> list[tuple[Array, Array]]:
        """Get a list of tuples of decay rates and collapse operators for each subsystem.

        Returns
        -------
        list[tuple[float, Array]]
            list of collapse operators

        """
        return self._hamiltonian.get_collapseops()

    def __get_ode_propagation_eom(self, time: Array) -> tuple[Array, list[Array]]:
        """
        Return the coherent and incoherent EOM parts seperately.
        Here the coherent part is the Hamiltonian as a function of time (w/o -1j)
        and the incoherent part is a list of collapse operators

        Parameters
        ----------
        time: Array
            Vector of time samples

        Returns
        -------
        tuple[Array, Array]
             Hamiltonian EOM ([t, N, N] matrix) and the `m` collapse operators ([m, N^2, N^2] matrix)
        """
        ham_eom = self._hamiltonian.get_matrix(time)
        rates_and_cols = self.get_collapseops()
        cols: list[Array] = [jnp.sqrt(rate) * col for rate, col in rates_and_cols]
        return -1j * ham_eom, cols

    def __create_hamiltonian_superop(self, t) -> Array | BCOO:
        """Create the Hamiltonian superoperator for one time point `t`."""
        identityop = jnp.eye(self._hamiltonian.dimension())
        ham = self._hamiltonian.get_matrix_one_time(t)
        superop = -1j * jnp.kron(identityop, ham) + 1j * jnp.kron(ham.T, identityop)
        if self.sparse_superop:
            return BCOO.fromdense(superop)
        return superop

    def __create_collapse_superop(self) -> Array | BCOO:
        """Create the superoperator due to the collapse part. This is time independent."""
        dim = self._hamiltonian.dimension()
        identityop = jnp.eye(dim)
        superop = jnp.zeros((dim**2, dim**2), dtype=jnp.float64)
        rates_and_cols = self._hamiltonian.get_collapseops()
        for rate, col in rates_and_cols:
            superop += rate * jnp.kron(col.conj(), col)
            superop -= rate / 2 * jnp.kron(jnp.matmul(col.T, col.conj()), identityop)
            superop -= rate / 2 * jnp.kron(identityop, jnp.matmul(col.conj().T, col))

        if self.sparse_superop:
            return BCOO.fromdense(superop)
        return superop

    def __create_lindbladian_superop(self, t) -> Array | BCOO:
        """Create the Lindbladian superoperator for one time point `t`."""
        hamSuperop = self.__create_hamiltonian_superop(t)
        colSuperop = self.__create_collapse_superop()
        return hamSuperop + colSuperop

    def get_matrix(self, time: Array):
        """
        Computes the right hand side of the Schrödinger equation without multiplying the state. Used for unitary
        solvers.

        Parameters
        ----------
        time : Array
            Vector of time samples

        Returns
        -------
        Array
            RHS with dimension [t, n, n]  with t: time, n: hilbert space
        """
        return self._get_matrix_method(time)

    @staticmethod
    @jit
    def __kron(A, B):
        return jnp.kron(A, B)

    def __create_hamiltonian_grad_superop(self, t):
        """Create the Gradient of Hamiltonian superoperator for one time point `t`."""
        identityop = jnp.eye(self._hamiltonian.dimension())
        ham_grad = self._hamiltonian.gradient_one_time(jnp.array([t]))
        term1 = -1j * vmap(self.__kron, in_axes=(None, 0))(identityop, ham_grad)
        term2 = 1j * vmap(self.__kron, in_axes=(0, None))(jnp.transpose(ham_grad, axes=(0, 2, 1)), identityop)
        superop = term1 + term2
        return superop

    def gradient(self, t) -> Array:
        """Compute the gradient of get_matrix."""
        if self.ode_propagation:
            grads = -1j * self._hamiltonian.gradient(t)
        else:
            grads = vmap(self.__create_hamiltonian_grad_superop)(t)
        return grads
