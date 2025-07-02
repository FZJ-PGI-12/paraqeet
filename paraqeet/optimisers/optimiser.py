"""Data class definition for the optimisation result object."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from paraqeet.file_logger import Logger
from paraqeet.measurement.measurement import Measurement
from paraqeet.optimisation_map import OptimisationMap


@dataclass(repr=False)
class OptimisationResult:
    """Data class for respresenting optimisation results.

    Attributes
    ----------
    STATUS_FINISHED : int
        The optimisation finished without a clear success or failure.
        This is used by algorithms that do not necessarily converge towards a
        solution.
    STATUS_SUCCESS : int
        The optimisation successfully found an optimum.
    STATUS_FAILED : int
        The optimisation failed to converge.
    status : int
        Indicates if the optimisation was successful.
        Should have one of the status constants as value.
    value : float
        The value at the best point of the optimised function.
    iterations : int
        The number of iterations during the optimisation.
    message : str | None, optional
        Any additional message from the optimisation algorithm.
        This can be an error message in case of failure.
    raw_result : Any | None, optional
        The raw result from the underlying algorithm.

    """

    # The optimisation finished without a clear success or failure.
    # This is used by algorithms that do not necessarily converge towards
    # a solution.
    STATUS_FINISHED = 0
    # The optimisation successfully found an optimum.
    STATUS_SUCCESS = 1
    # The optimisation failed to converge.
    STATUS_FAILED = 2

    # Indicates if the optimisation was successful.
    # Should have one of the status constants as value.
    status: int
    # The value at the best point of the optimised function.
    value: float
    # The number of iterations during the optimisation.
    iterations: int
    # Any additional message from the optimisation algorithm.
    # This can be an error message in case of failure.
    message: str | None = None
    # The raw result from the underlying algorithm.
    raw_result: Any | None = None

    def __repr__(self):
        """Magic method for human-readable printable representation.

        Represents the Optimiser object as a dictionary with status,
        value, and iterations. If a message has been added, adds that
        too the dict too.

        """
        asDict = {
            "status": self.status,
            "value": self.value,
            "iterations": self.iterations,
        }
        if self.message:
            asDict["message"] = self.message

        return str(asDict)


class Optimiser:
    """Base class for all classes that implement an optimisation algorithm.

    The class accepts a list of optimisable parameters from the lower layers
    which shall be optimised in order to minimise the given measure.

    Parameters
    ----------
    measure : Measurement
        Implementation of the Measurement class that measures the observable
        to be minimised.
    optimisables : OptimisationMap
        An optimisation map containing all parameters that can be optimised.
        If none, an empty map will be created to which the parameters can
        be added later used.
    logger : FileLogger
        The file logger object.

    """

    _measure: Measurement
    _optimisables: OptimisationMap
    __opt_idxs: list[int]
    __logger: Logger | None

    def __init__(
        self,
        measure: Measurement,
        optimisables: OptimisationMap,
        logger: Logger | None = None,
    ):
        self._measure = measure
        self._logger = logger
        self.optimisables = optimisables

    @property
    def logger(self) -> Logger | None:
        """Returns the current logger that is being used by this optimiser, or None if no logger was set yet."""
        return self._logger

    @logger.setter
    def logger(self, logger: Logger):
        """Set the logger for the optimiser object.

        Parameters
        ----------
        logger : Logger
            Logger object to be set as the logger for the system.
        """
        self._logger = logger

    @property
    def optimisables(self) -> OptimisationMap:
        """Return the optimisation map that this optimiser uses.

        Parameters that can be optimised need to be added to this map.

        Returns
        -------
        paraqeet.optimisation_map
            Returns the optimisation map that this optimiser uses.

        """
        return self._optimisables

    @optimisables.setter
    def optimisables(self, opt: OptimisationMap) -> None:
        """Set optimisable options (via Map).

        Registers optimisables and their length to keep track of vector
        and matrix valued parameters.

        Parameters
        ----------
        opt : OptimisationMap
            Takes in the optimisables to set parameters.

        """
        self._optimisables = opt

    @abstractmethod
    def optimise(self) -> OptimisationResult:
        """Perform the actual optimisation.

        Depending on the implementation, this function might take a long
        time and might need several calls to the underlying layers.
        The returned object contains some information about the result.
        The result will include the raw result of the underlying algorithm
        for more information.

        Returns
        -------
        OptimisationResult
            Result of optimization via the OptimisationResult object.
            (status, value, iterations and the raw result)

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    def _build_optimisable_index_list(self):
        """Build the optimisable index list.

        Register optimisables and their length to keep track of vector
        and matrix valued parameters.

        """
        params = self._optimisables.get_all_parameters()
        self._opt_idxs = []
        index = 0
        for qty in params:
            index += qty.get_length()
            self._opt_idxs.append(index)
