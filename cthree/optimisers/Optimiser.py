from abc import abstractmethod
from typing import List

from cthree.OptimisationMap import OptimisationMap
from cthree.measurement.Measurement import Measurement
from cthree.FileLogger import Logger


class OptimisationResult:

    STATUS_FINISHED = 'finished'
    """
    The optimisation finished without a clear success or failure. This is used by algorithms that do not necessarily
    converge towards a solution.
    """
    STATUS_SUCCESS = 'success'
    """ The optimisation successfully found an optimum. """
    STATUS_FAILED = 'failed'
    """ The optimisation failed to converge. """

    status: int
    """ Indicates if the optimisation was successful. Should have one of the status constants as value. """
    value: float
    """ The value at the best point of the optimised function. """
    iterations: int
    """ The number of iterations during the optimisation. """
    message: str | None
    """ Any additional message from the optimisation algorithm. This can be an error message in case of failure. """

    def __init__(self, status: int, value: float, iterations: int, message: str | None = None) -> None:
        self.status = status
        self.value = value
        self.iterations = iterations
        self.message = message

    def __repr__(self):
        asDict = {
            'status': self.status,
            'value': self.value,
            'iterations': self.iterations,
        }
        if self.message:
            asDict['message'] = self.message

        return str(asDict)


class Optimiser:
    """
    Base class for all classes that implement an optimisation algorithm. The class accepts a list of optimisable
    parameters from the lower layers which shall be optimised in order to minimise the given measure.

    Args:
        measure: implementation of the Measurement class that measures the observable to be minimised
        optimisables: An optimisation map containing all parameters that can be optimised. If none, an empty map will
         be created to which the parameters can be added later
        used.
    """

    _measure: Measurement
    _optimisables: OptimisationMap
    _rawResult = None
    __opt_idxs: List[int]
    __logger: Logger

    def __init__(
        self, measure: Measurement, optimisables: OptimisationMap, logger: Logger = None
    ):
        self._measure = measure
        self._logger = logger
        self.setOptimisables(optimisables)

    def setLogger(self, logger: Logger):
        self._logger = logger

    def getOptimisables(self) -> OptimisationMap:
        """
        Returns the optimisation map that this optimiser uses. Parameters that can be optimised need to be added to this
        map.
        """
        return self._optimisables

    def setOptimisables(self, opt: OptimisationMap) -> None:
        """
        Registers optimisables and their length to keep track of vector and matrix valued parameters.
        """
        self._optimisables = opt

    @abstractmethod
    def optimise(self) -> OptimisationResult:
        """
        Performs the actual optimisation. Depending on the implementation, this function might take a long time and
        might need several calls to the underlying layers. The returned object contains some information about the
        result. The raw result of the underlying algorithm can be retrieved from the getRawResult method once the
        optimisation is complete.
        """
        raise NotImplementedError()

    def getRawResult(self):
        """
        Returns the raw response of the underlying algorithm from the last optimisation. The format depends on the
        implementation.
        """
        return self._rawResult

    def _buildOptimisableIndexList(self):
        """
        Register optimisables and their length to keep track of vector and matrix valued parameters.
        """
        params = self._optimisables.getAllParameters()
        self._opt_idxs = []
        index = 0
        for qty in params:
            index += qty.getLength()
            self._opt_idxs.append(index)
