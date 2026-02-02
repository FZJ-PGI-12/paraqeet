"""Data class definition for the optimization result object."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from paraqeet.file_logger import Logger
from paraqeet.measurement.measurement import Measurement
from paraqeet.optimization_map import OptimizationMap
from paraqeet.quantity import Array


@dataclass(repr=False)
class OptimizationResult:
    """Data class for respresenting optimization results.

    Attributes
    ----------
    STATUS_FINISHED : int
        The optimization finished without a clear success or failure.
        This is used by algorithms that do not necessarily converge towards a
        solution.
    STATUS_SUCCESS : int
        The optimization successfully found an optimum.
    STATUS_FAILED : int
        The optimization failed to converge.
    status : int
        Indicates if the optimization was successful.
        Should have one of the status constants as value.
    value : float
        The value at the best point of the optimized function.
    iterations : int
        The number of iterations during the optimization.
    message : str | None, optional
        Any additional message from the optimization algorithm.
        This can be an error message in case of failure.
    raw_result : Any | None, optional
        The raw result from the underlying algorithm.

    """

    # The optimization finished without a clear success or failure.
    # This is used by algorithms that do not necessarily converge towards
    # a solution.
    STATUS_FINISHED = 0
    # The optimization successfully found an optimum.
    STATUS_SUCCESS = 1
    # The optimization failed to converge.
    STATUS_FAILED = 2

    # Indicates if the optimization was successful.
    # Should have one of the status constants as value.
    status: int
    # The value at the best point of the optimized function.
    value: float
    # The number of iterations during the optimization.
    iterations: int
    # Any additional message from the optimization algorithm.
    # This can be an error message in case of failure.
    message: str | None = None
    # The raw result from the underlying algorithm.
    raw_result: Any | None = None

    def __repr__(self):
        """Magic method for human-readable printable representation.

        Represents the Optimizer object as a dictionary with status,
        value, and iterations. If a message has been added, adds that
        too the dict too.

        """
        as_dict = {
            "status": self.status,
            "value": self.value,
            "iterations": self.iterations,
        }
        if self.message:
            as_dict["message"] = self.message

        return str(as_dict)


class Optimizer(ABC):
    """Base class for all classes that implement an optimization algorithm.

    The class accepts a list of optimizable parameters from the lower layers
    which shall be optimized in order to minimise the given measure.

    Parameters
    ----------
    measure: Measurement
        Implementation of the Measurement class that measures the observable
        to be minimised.
    optimizables: OptimizationMap
        An optimization map containing all parameters that can be optimized.
        If none, an empty map will be created to which the parameters can
        be added later used.
    logger: FileLogger
        The file logger object.

    """

    _measure: Measurement
    _optimization_map: OptimizationMap
    _opt_idxs: list[int]
    _logger: Logger | None
    _times: Array

    def __init__(
        self,
        measure: Measurement,
        optimization_map: OptimizationMap,
        logger: Logger | None = None,
    ):
        self._measure = measure
        self._logger = logger
        self.optimization_map = optimization_map

    @property
    def logger(self) -> Logger | None:
        """Returns the current logger that is being used by this optimizer, or None if no logger was set yet."""
        return self._logger

    @logger.setter
    def logger(self, logger: Logger):
        """Set the logger for the optimizer object.

        Parameters
        ----------
        logger : Logger
            Logger object to be set as the logger for the system.
        """
        self._logger = logger

    @property
    def optimization_map(self) -> OptimizationMap:
        """Return the optimization map that this optimizer uses.

        Parameters that can be optimized need to be added to this map.

        Returns
        -------
        paraqeet.optimization_map
            Returns the optimization map that this optimizer uses.

        """
        return self._optimization_map

    @optimization_map.setter
    def optimization_map(self, opt: OptimizationMap) -> None:
        """Set an optimization_map.

        Registers optimizables and their length to keep track of vector
        and matrix valued parameters.

        Parameters
        ----------
        opt: OptimizationMap
            Takes in the optimizables to set parameters.

        """
        self._optimization_map = opt

    @abstractmethod
    def optimize(self, times: Array | float) -> OptimizationResult:
        """Perform the actual optimization.

        Depending on the implementation, this function might take a long
        time and might need several calls to the underlying layers.
        The returned object contains some information about the result.
        The result will include the raw result of the underlying algorithm
        for more information.

        Returns
        -------
        OptimizationResult
            Result of optimization via the OptimizationResult object.
            (status, value, iterations and the raw result)

        """
        pass

    def _build_optimizable_index_list(self):
        """Build the optimizable index list.

        Register optimizables and their length to keep track of vector
        and matrix valued parameters.

        """
        params = self._optimization_map.get_all_parameters()
        self._opt_idxs = []
        index = 0
        for qty in params:
            index += qty.get_length()
            self._opt_idxs.append(index)
