"""Class definition for a mixed state transfer fidelity model."""

from paraqeet.quantity import Array
import jax.numpy as jnp
import jax.scipy.linalg as sclin

from paraqeet.exceptions import IncompatibleLayersException
from paraqeet.measurement.measurement import Measurement
from paraqeet.propagation.propagation import Propagation
from paraqeet.quantity import Quantity


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

    __target_state: Array
    __target_state_sqrt: Array
    __propagation: Propagation
    __times: Array

    def __init__(
        self,
        propagation: Propagation,
        targetState: Array,
        times: Array,
    ):
        self.__propagation = propagation
        self.__target_state = targetState
        self.__times = times

        # store the sqrt of the density matrix to simplify the measurement
        self.__target_state_sqrt = sclin.sqrtm(self.__target_state)

    def get_parameters(self) -> list[Quantity]:
        """Returns an empty list."""
        return []

    def measure(self) -> Array:
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
        state = self.__propagation.propagate(self.__times)[-1]
        state = self._preprocess_matrix(state)
        if state.shape != self.__target_state.shape:
            raise IncompatibleLayersException(
                f"Need a state vector of size {self.__target_state.shape}"
                "for the state transfer fidelity, "
                "but got shape {state.shape}"
            )

        # density matrix
        product = self.__target_state_sqrt @ state @ self.__target_state_sqrt
        return jnp.abs(jnp.trace(sclin.sqrtm(product))) ** 2
