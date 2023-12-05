from typing import Dict, List, Set

from cthree.Optimisable import Optimisable
from cthree.Quantity import Quantity


class OptimisationMap:
    """
    Utility class that collects all parameters that shall be considered during optimisation and associates them will
    the corresponding Optimisable interface. With this class, Quantities can be traced back to the Optimisable to which
    they belong. Before optimisation, an instance of this class needs to be filled and passed to the optimiser.
    """
    __optimisableToParameterMap: Dict[Optimisable, List[Quantity]]

    def __init__(self):
        self.__optimisableToParameterMap = {}

    def add(self, optimisable: Optimisable, optimisableQuantites: List[Quantity] = None):
        """
        Adds an optimisable object and a list of its quantities to the map. The list contains all parameters
        of the optimisable object that shall be considered during the optimisation. If the list is empty, all parameters
        of the class will be used. If the object was already added, the list of quantities will be overwritten.

        :param optimisable:
        :param optimisableQuantites:
        :return:
        """
        params = optimisableQuantites or optimisable.getParameters()
        self.__optimisableToParameterMap[optimisable] = params

    def getOptimisables(self) -> Set[Optimisable]:
        """
        Returns all optimisable objects that were added to this map.
        """
        return set(self.__optimisableToParameterMap.keys())

    def getParameters(self, optimisable: Optimisable) -> List[Quantity] | None:
        """
        Returns all quantities associated with the optimisable object that were added to this map.

        :param optimisable:
        :return: the list of parameters or None if the optimisable has not been added yet
        """

    def getAllParameters(self) -> List[Quantity]:
        """
        Returns all parameters that were added to this map for any optimisable object.
        """
        quantities = []
        for params in self.__optimisableToParameterMap.values():
            quantities += params
        return quantities
