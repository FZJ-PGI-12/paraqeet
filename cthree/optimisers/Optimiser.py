from abc import abstractmethod
from typing import List

from cthree.OptimisationMap import OptimisationMap
from cthree.measurement.Measurement import Measurement
from cthree.FileLogger import Logger


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
    def optimise(self):
        """
        Performs the actual optimisation. Depending on the implementation, this function might take a long time and
        might need several calls to the underlying layers.
        """
        raise NotImplementedError()
