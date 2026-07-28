"""Lindblad master equation of motion for open quantum systems."""

from collections.abc import Callable
from typing import override

import jax.numpy as jnp
from jax import vmap

from paraqeet.hamiltonian.equation_of_motion import EquationOfMotion
from paraqeet.quantity import Array


class MasterEquation(EquationOfMotion):
    """Model of an open quantum system, defined by the Hamiltonian and jump operators.

    Its dynamics is given by the Lindblad master equation :cite:p:`lindblad1976generators,manzano2020short`.

    Defaults to returning the Lindblad superoperator. For ODE based methods use the
    ``get_eom_ode_propagation`` and ``get_eom_gradient_ode_propagation`` methods.
    """

    _jump_operators: list[Array]
    _total_dimension: int

    def __init__(
        self,
        hamiltonian_func: Callable[[Array], Array],
        hamiltonian_gradient_func: Callable[[Array], Array],
        jump_operators: list[Array],
    ) -> None:
        """
        Args:
            hamiltonian_func: Hamiltonian as a function of time.
            hamiltonian_gradient_func: Hamiltonian gradients as a function of time.
            jump_operators: List of jump operators (each multiplied by the sqrt
                of their corresponding decay rates).
        """
        super().__init__(hamiltonian_func, hamiltonian_gradient_func)
        self._jump_operators = jump_operators
        self._total_dimension = hamiltonian_func(jnp.array([0.0])).shape[1]

    @property
    def jump_operators(self) -> list[Array]:
        """Return a list of jump operators (each multiplied by the sqrt of their corresponding decay rates).

        Returns:
            List of jump operators.
        """
        return self._jump_operators

    @jump_operators.setter
    def jump_operators(self, jump_ops: list[Array]) -> None:
        """Set a list of jump operators (each multiplied by the sqrt of their corresponding decay rates)."""
        self._jump_operators = jump_ops

    def _create_hamiltonian_superop(self, t: Array) -> Array:
        """Create the Hamiltonian superoperator for one time point ``t``."""
        identityop = jnp.eye(self._total_dimension)
        ham = self._hamiltonian_func(jnp.array(t, ndmin=1)).squeeze(axis=0)
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

    def _create_lindbladian_superop(self, t: Array) -> Array:
        """Create the Lindbladian superoperator for one time point ``t``."""
        ham_super_op = vmap(self._create_hamiltonian_superop)(t)
        col_super_op = self._create_jump_superop()
        return ham_super_op + col_super_op

    def _create_hamiltonian_grad_superop(self, timestep: float) -> Array:
        """Create the Gradient of Hamiltonian superoperator for one time point ``timestep``."""
        identityop = jnp.eye(self._total_dimension)
        ham_grad = self._hamiltonian_gradient_func(jnp.array(timestep, ndmin=1))
        ham_grad = ham_grad.squeeze(axis=0)
        term1 = -1j * vmap(jnp.kron, in_axes=(None, 0))(identityop, ham_grad)
        term2 = 1j * vmap(jnp.kron, in_axes=(0, None))(jnp.transpose(ham_grad, axes=(0, 2, 1)), identityop)
        superop = term1 + term2
        return superop

    def get_eom_ode_propagation(self, times: Array) -> Array:
        """Return EOM for ODE propagation methods.

        Return the coherent part of the EOM.
        Here the coherent part is the Hamiltonian as a function of time (w/o -1j).
        The incoherent part is a list of jump operators and can be obtained by the
        ``jump_operators`` attribute.

        Args:
            times: Vector of time samples.

        Returns:
            Hamiltonian EOM ([t, N, N] matrix).
        """
        ham_eom = self._hamiltonian_func(times)
        return -1j * ham_eom

    def get_eom_gradient_ode_propagation(self, times: Array) -> Array:
        """Return the gradient of the EOM for ODE propagation methods.

        Return the coherent part of the EOM, i.e., the Hamiltonian and its gradient.
        The incoherent part is a list of jump operators and can be obtained by the
        ``jump_operators`` attribute.

        Args:
            times: Vector of time samples.

        Returns:
            Hamiltonian EOM gradient ([t, N, N] matrix).
        """
        grads = self._hamiltonian_gradient_func(times)
        return -1j * grads

    @override
    def get_value(self, times: Array) -> Array:
        """Return the Lindblad superoperator.

        Args:
            times: Vector of time samples.

        Returns:
            RHS with dimension [t, N^2, N^2] with t: time, N: Hilbert space.
        """
        return self._create_lindbladian_superop(times)

    @override
    def get_gradient(self, times: Array) -> Array:
        """Return the gradient of the Lindbladian superoperator.

        Args:
            times: Vector of time samples.

        Returns:
            RHS with dimension [t, N^2, N^2] with t: time, N: Hilbert space.
        """
        grads: Array = vmap(self._create_hamiltonian_grad_superop)(times)  # type: ignore
        return grads
