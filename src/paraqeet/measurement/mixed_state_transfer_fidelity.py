"""Class definition for a mixed state transfer fidelity model."""

from collections.abc import Callable
from typing import override

import jax.numpy as jnp
import jax.scipy.linalg as sclin

from paraqeet.exceptions import IncompatibleLayersException
from paraqeet.measurement.measurement import Measurement
from paraqeet.quantity import Array, Float


class MixedStateTransferFidelity(Measurement):
    """Mixed state transfer fidelity measurement model.

    Fidelity measure that compares the overlap of the initial
    and final state of density matrices.
    Note: this implementation is still very inaccurate.
    """

    _target_state: Array
    _target_state_sqrt: Array
    _propagation_func: Callable[[Array], Array]

    def __init__(
        self,
        propagation_func: Callable[[Array], Array],
        target_state: Array,
    ):
        """
        Args:
            propagation_func: Function that evaluates the propagation of some
                initial state. Expected to be of the form
                ``func(t: Array) -> states: Array``.
            target_state: Target state.
        """
        self._propagation_func = propagation_func
        self._target_state = target_state

        # store the sqrt of the density matrix to simplify the measurement
        self._target_state_sqrt = sclin.sqrtm(self._target_state)

    @override
    def measure(self, times: Array) -> Array | Float:
        """Measure overlap between initial and final state of density matrices.

        Returns:
            Overlap between initial and final state of density matrices.

        Raises:
            IncompatibleLayersException: If required vector shape is not
                received.
        """
        state = self._propagation_func(times)[-1]
        if state.shape != self._target_state.shape:
            raise IncompatibleLayersException(
                f"Need a state vector of size {self._target_state.shape}"
                "for the state transfer fidelity, "
                "but got shape {state.shape}"
            )

        # density matrix
        product = self._target_state_sqrt @ state @ self._target_state_sqrt
        return jnp.abs(jnp.trace(sclin.sqrtm(product))) ** 2
