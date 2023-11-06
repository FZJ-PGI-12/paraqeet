from abc import abstractmethod
from typing import List, Optional

from cthree.Quantity import Quantity
from cthree.measurement.Measurement import Measurement


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

    def __init__(
        self, measure: Measurement, optimisables: Optional[List[Quantity]] = None
    ):
        self._measure = measure
        self.setOptimisables(optimisables or measure.getParameters())

    def setOptimisables(self, opt: List[Quantity]) -> None:
        """
        Registers optimisables and their length to keep track of vector and matrix valued parameters.
        """
        self._optimisables = opt
        self.opt_idxs = []
        index = 0
        for qty in opt:
            index += qty.getLength()
            self.opt_idxs.append(index)

    @abstractmethod
    def optimise(self):
        """
        Performs the actual optimisation. Depending on the implementation, this function might take a long time and
        might need several calls to the underlying layers.
        """
        raise NotImplementedError()
