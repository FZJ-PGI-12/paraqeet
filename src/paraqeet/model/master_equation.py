"""Class definition of an open system."""

from collections.abc import Callable

import jax.numpy as jnp
from jax import vmap

from paraqeet.model.equation_of_motion import EquationOfMotion
from paraqeet.quantity import Array


class MasterEquation(EquationOfMotion):
    """
    Model of an open quantum system, defined by the Hamiltonian and jump operators.
    Its dynamics given by the Lindblad master equation.

    Defaults to returning the Lindblad superoperator. For ODE based methods use the
    `get_eom_ode_propagation` and `get_eom_and_gradient_ode_propagation` methods.

    Parameters
    ----------
    hamiltonian_func : Callable[[Array], Array]
        Hamiltonian as a function of time.
    hamiltonian_and_gradient_func: Callable[[Array], tuple[Array, Array]]
        Hamiltonian, Hamiltonian gradients as a function of time.
    jump_operators: list[Array]
        Jump operators present in the system (multiplied by the sqrt of corresponding decay rates).
    """

    _jump_operators: list[Array]
    _total_dimension: int

    def __init__(
        self,
        hamiltonian_func: Callable[[Array], Array],
        hamiltonian_and_gradient_func: Callable[[Array], tuple[Array, Array]],
        jump_operators: list[Array],
    ):
        super().__init__(hamiltonian_func, hamiltonian_and_gradient_func)
        self._jump_operators = jump_operators
        self._total_dimension = hamiltonian_func(jnp.array([0.0])).shape[1]

    @property
    def jump_operators(self) -> list[Array]:
        """Return a list of jump operators (each multiplied by the sqrt of their corresponding decay rates).

        Returns
        -------
        list[Array]
            list of jump operators

        """
        return self._jump_operators

    @jump_operators.setter
    def jump_operators(self, jump_ops: list[Array]):
        """Set a list of jump operators (each multiplied by the sqrt of their corresponding decay rates)."""
        self._jump_operators = jump_ops

    def _create_hamiltonian_superop(self, t) -> Array:
        """Create the Hamiltonian superoperator for one time point `t`."""
        identityop = jnp.eye(self._total_dimension)
        ham = self._hamiltonian_func(t)
        superop = -1j * jnp.kron(identityop, ham) + 1j * jnp.kron(ham.T, identityop)
        return superop

    def _create_jump_superop(self) -> Array:
        """Create the superoperator due to the jump part. This is time independent."""
        identityop = jnp.eye(self._total_dimension)
        superop = jnp.zeros((self._total_dimension**2, self._total_dimension**2), dtype=jnp.float64)
        for jump_op in self.jump_operators:
            superop += jnp.kron(jump_op.conj(), jump_op)
            superop -= jnp.kron(jnp.matmul(jump_op.T, jump_op.conj()), identityop) / 2
            superop -= jnp.kron(identityop, jnp.matmul(jump_op.conj().T, jump_op)) / 2
        return superop

    def _create_lindbladian_superop(self, t) -> Array:
        """Create the Lindbladian superoperator for one time point `t`."""
        ham_super_op = self._create_hamiltonian_superop(t)
        col_super_op = self._create_jump_superop()
        return ham_super_op + col_super_op

    def _create_hamiltonian_grad_superop(self, timestep: float):
        """Create the Gradient of Hamiltonian superoperator for one time point `timestep`."""
        identityop = jnp.eye(self._total_dimension)
        ham_grad = self._hamiltonian_and_gradient_func(timestep)
        term1 = -1j * vmap(jnp.kron, in_axes=(None, 0))(identityop, ham_grad)
        term2 = 1j * vmap(jnp.kron, in_axes=(0, None))(jnp.transpose(ham_grad, axes=(0, 2, 1)), identityop)
        superop = term1 + term2
        return superop

    def get_eom_ode_propagation(self, times: Array) -> tuple[Array, list[Array]]:
        """Return EOM for ODE propagation methods.

        Return the coherent and incoherent EOM parts seperately.
        Here the coherent part is the Hamiltonian as a function of time (w/o -1j)
        and the incoherent part is a list of jump operators

        Parameters
        ----------
        times: Array
            Vector of time samples

        Returns
        -------
        tuple[Array, Array]
             Hamiltonian EOM ([t, N, N] matrix) and the `m` jump operators ([m, N^2, N^2] matrix)
        """
        ham_eom = self._hamiltonian_func(times)
        return -1j * ham_eom, self.jump_operators

    def get_eom_and_gradient_ode_propagation(self, times: Array) -> tuple[Array, Array]:
        """Return EOM for ODE propagation methods.

        Return the coherent and incoherent EOM parts seperately.
        Here the coherent part is the Hamiltonian as a function of time (w/o -1j)
        and the incoherent part is a list of jump operators

        Parameters
        ----------
        times: Array
            Vector of time samples

        Returns
        -------
        tuple[Array, Array]
             Hamiltonian EOM ([t, N, N] matrix) and the `m` jump operators ([m, N^2, N^2] matrix)
        """
        ham_eom, grads = self._hamiltonian_and_gradient_func(times)
        return -1j * ham_eom, -1j * grads

    def get_value(self, times: Array):
        """Return the Lindblad superoperator.

        Parameters
        ----------
        times: Array
            Vector of time samples

        Returns
        -------
        Array
            RHS with dimension [t, N^2, N^2]  with t: time, N: hilbert space
        """
        return vmap(self._create_lindbladian_superop)(times)

    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array]:
        """Return the Lindblad superoperator and its gradient.

        Parameters
        ----------
        times: Array
            Vector of time samples

        Returns
        -------
        Array
            RHS with dimension [t, N^2, N^2]  with t: time, N: hilbert space
        """
        eom = vmap(self._create_lindbladian_superop)(times)
        # ignoring mypy due to vmap
        grads = vmap(self._create_hamiltonian_grad_superop)(times)  # type: ignore
        return eom, grads
