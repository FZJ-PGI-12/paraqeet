"""Class definition for a mixed state transfer fidelity model."""

import jax.numpy as jnp
import jax.scipy.linalg as sclin

from paraqeet.exceptions import IncompatibleLayersException
from paraqeet.measurement.measurement import Measurement
from paraqeet.propagation.propagation import Propagation
from paraqeet.quantity import Array


class MixedStateTransferFidelity(Measurement):
    """Mixed state transfer fidelity measurement model.

    Fidelity measure that compares the overlap of the initial
    and final state of density matrices.
    Note: this implementation is still very inaccurate.

    Parameters
    ----------
    propagation : Propagation
        Abstract base class for any implementation that can solve
        the equation of motion.
    targetState : Array
        Final state of the density matrices.
    times : Array
        One-dimensional vector of timestamps.

    """

    _target_state: Array
    _target_state_sqrt: Array
    _propagation: Propagation

    def __init__(
        self,
        propagation: Propagation,
        targetState: Array,
    ):
        self._propagation = propagation
        self._target_state = targetState

        # store the sqrt of the density matrix to simplify the measurement
        self._target_state_sqrt = sclin.sqrtm(self._target_state)

    def measure(self, times: Array) -> Array | float:
        """Measure overlap between initial and final state of density matrices.

        Returns
        -------
        Array
            Overlap between initial and final state of density matrices.

        Raises
        ------
        IncompatibleLayersException
            Raises an exception if required vector shape is not received.

        """
        state = self._propagation.propagate(times)[-1]
        if state.shape != self._target_state.shape:
            raise IncompatibleLayersException(
                f"Need a state vector of size {self._target_state.shape}"
                "for the state transfer fidelity, "
                "but got shape {state.shape}"
            )

        # density matrix
        product = self._target_state_sqrt @ state @ self._target_state_sqrt
        return jnp.abs(jnp.trace(sclin.sqrtm(product))) ** 2
