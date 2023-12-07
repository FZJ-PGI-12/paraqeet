from abc import abstractmethod
from typing import List, Optional

from cthree.Quantity import Quantity
from cthree.measurement.Measurement import Measurement
from cthree.FileLogger import Logger


class Optimiser:
    """
    Base class for all classes that implement an optimisation algorithm. The class accepts a list of optimisable
    parameters from the lower layers which shall be optimised in order to minimise the given measure.

    Args:
        measure: implementation of the Measurement class that measures the observable to be minimised
        optimisables: A list of parameters that can be optimised. If none, only the parameters of the measure will be
        used.
    """

    _measure: Measurement
    _optimisables: List[Quantity]
    __opt_idxs: List[int]
    __logger: Logger

    def __init__(
        self,
        measure: Measurement,
        optimisables: Optional[List[Quantity]] = None,
        logger: Logger = None
    ):
        self._measure = measure
        self._logger = logger
        self.setOptimisables(optimisables or measure.getParameters())

    def setLogger(self, logger: Logger):
        self._logger = logger

    def setOptimisables(self, opt: List[Quantity]) -> None:
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
