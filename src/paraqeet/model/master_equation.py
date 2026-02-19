"""Class definition of an open system."""

from collections.abc import Callable

import jax.numpy as jnp
from jax import jit, vmap
from jax.experimental.sparse import BCOO

from paraqeet.differentiable import Differentiable
from paraqeet.exceptions import ConfigurationException
from paraqeet.model.equation_of_motion import EquationOfMotion
from paraqeet.model.solvable import Solvable
from paraqeet.model.system import OpenSystem
from paraqeet.quantity import Array


class MasterEquation(EquationOfMotion):
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
        If `true` then `get_value` method returns list of Hamiltonian (with time) and collapse operator.
        Else returns Lindblad superoperator.
    """

    _ode_propagation: bool
    _sparse_superop: bool
    _get_value_method: Callable
    _solvable: Solvable

    def __init__(
        self,
        solvable: Solvable,
        sparse_superop: bool = False,
        ode_propagation: bool = False,
    ):
        self._solvable = solvable
        self._sparse_superop = sparse_superop
        self.ode_propagation = ode_propagation

    @property
    def sparse_superop(self) -> bool:
        """Flag to store superoperators as sparse matrices.

        Returns
        -------
        bool
            Flag to store sparse matrices.
        """
        return self._sparse_superop

    @sparse_superop.setter
    def sparse_superop(self, sparse_superop: bool) -> None:
        self._sparse_superop = sparse_superop

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
            self._get_value_method = self._get_ode_propagation_eom
        else:
            self._get_value_method = vmap(self._create_lindbladian_superop)

    def get_collapseops(self) -> list[tuple[Array, Array]]:
        """Get a list of tuples of decay rates and collapse operators for each subsystem.

        Returns
        -------
        list[tuple[float, Array]]
            list of collapse operators

        """
        sol = self._solvable
        if isinstance(sol, OpenSystem):
            coll_ops = sol.get_collapseops()
        else:
            raise ConfigurationException(f"{sol} is not an Open System.")
        return coll_ops

    def _get_ode_propagation_eom(self, times: Array) -> tuple[Array, list[Array]]:
        """
        Return the coherent and incoherent EOM parts seperately.
        Here the coherent part is the Hamiltonian as a function of time (w/o -1j)
        and the incoherent part is a list of collapse operators

        Parameters
        ----------
        times: Array
            Vector of time samples

        Returns
        -------
        tuple[Array, Array]
             Hamiltonian EOM ([t, N, N] matrix) and the `m` collapse operators ([m, N^2, N^2] matrix)
        """
        ham_eom = self._solvable.get_hamiltonian(times)
        rates_and_cols = self.get_collapseops()
        cols: list[Array] = [jnp.sqrt(rate) * col for rate, col in rates_and_cols]
        return -1j * ham_eom, cols

    def _create_hamiltonian_superop(self, t) -> Array | BCOO:
        """Create the Hamiltonian superoperator for one time point `t`."""
        identityop = jnp.eye(self._solvable.dimension())
        ham = self._solvable.get_value_at_timestep(t)
        superop = -1j * jnp.kron(identityop, ham) + 1j * jnp.kron(ham.T, identityop)
        if self.sparse_superop:
            return BCOO.fromdense(superop)
        return superop

    def _create_collapse_superop(self) -> Array | BCOO:
        """Create the superoperator due to the collapse part. This is time independent."""
        dim = self._solvable.dimension()
        identityop = jnp.eye(dim)
        superop = jnp.zeros((dim**2, dim**2), dtype=jnp.float64)
        rates_and_cols = self.get_collapseops()
        for rate, col in rates_and_cols:
            superop += rate * jnp.kron(col.conj(), col)
            superop -= rate * jnp.kron(jnp.matmul(col.T, col.conj()), identityop) / 2
            superop -= rate * jnp.kron(identityop, jnp.matmul(col.conj().T, col)) / 2

        if self.sparse_superop:
            return BCOO.fromdense(superop)
        return superop

    def _create_lindbladian_superop(self, t) -> Array | BCOO:
        """Create the Lindbladian superoperator for one time point `t`."""
        ham_super_op = self._create_hamiltonian_superop(t)
        col_super_op = self._create_collapse_superop()
        return ham_super_op + col_super_op

    # TODO: check the times-Array: internally a method might be called which expects only one timestep
    def get_value(self, times: Array):
        """
        Computes the right hand side of the Schrödinger equation without multiplying the state.
        Used for unitary solvers.

        Parameters
        ----------
        times: Array
            Vector of time samples

        Returns
        -------
        Array
            RHS with dimension [t, n, n]  with t: time, n: hilbert space
        """
        # TODO: in case the matrix_method is _create_lindbladian_superop, only one timestep is expected!
        return self._get_value_method(times)

    @staticmethod
    @jit
    def _kron(A, B):
        return jnp.kron(A, B)

    def _create_hamiltonian_grad_superop(self, timestep: float):
        """Create the Gradient of Hamiltonian superoperator for one time point `timestep`."""
        identityop = jnp.eye(self._solvable.dimension())
        ham_grad = self._solvable.get_gradient_at_timestep(timestep)
        term1 = -1j * vmap(MasterEquation._kron, in_axes=(None, 0))(identityop, ham_grad)
        term2 = 1j * vmap(MasterEquation._kron, in_axes=(0, None))(jnp.transpose(ham_grad, axes=(0, 2, 1)), identityop)
        superop = term1 + term2
        return superop

    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array]:
        """Compute the gradient of get_value."""
        # TODO: What to do if Hamiltonian is not differentiable?
        if isinstance(self._solvable, Differentiable):
            if self.ode_propagation:
                # TODO: Can the eom be obtained without calling the get_value method?
                _, grads = self._solvable.get_value_and_gradient(times)
                eom = self._get_value_method(times)
                grads = -1j * grads
            else:
                eom = vmap(self._create_lindbladian_superop)(times)
                # TODO: times is an Array but float is expected
                # ignoring mypy due to vmap
                grads = vmap(self._create_hamiltonian_grad_superop)(times)  # type: ignore
        return eom, grads
