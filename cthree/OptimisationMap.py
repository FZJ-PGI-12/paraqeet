from typing import Dict, List, Set

from cthree.Optimisable import Optimisable
from cthree.Quantity import Quantity


class OptimisationMap:
    """
    Utility class that collects all parameters that shall be considered during optimisation and associates them with
    the corresponding Optimisable interface. With this class, Quantities can be traced back to the Optimisable to which
    they belong. Before optimisation, an instance of this class needs to be filled and passed to the optimiser.
    """

    __optimisableToParameterMap: Dict[Optimisable, List[Quantity]]

    def __init__(self):
        self.__optimisableToParameterMap = {}

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """
        Human readable representation of the parameters set to optimise.
        """
        om_str = ""
        for key, val in self.__optimisableToParameterMap.items():
            om_str += f"==== {key} ====\n"
            om_str += str(val)
            om_str += "\n\n"
        return om_str

    def add(
        self, optimisable: Optimisable, optimisableQuantites: List[Quantity] = None
    ):
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
        return self.__optimisableToParameterMap[optimisable]

    def getAllParameters(self) -> List[Quantity]:
        """
        Returns all parameters that were added to this map for any optimisable object.
        """
        quantities = []
        for params in self.__optimisableToParameterMap.values():
            quantities.extend(params)
        return quantities

    def registerParamsWithOptimisables(self) -> None:
        """
        Utility function that synchronises the list of parameters with each optimisable class. This needs to be called
        by the optimiser before gradient based optimisation to tell the layers which gradients to compute.
        :return:
        """
        for optimisable, params in self.__optimisableToParameterMap.items():
            optimisable.setOptimisableParameters(params)

    def filterParameters(self, filterFunction) -> None:
        """
        Updates the list of parameters for all Optimisables in this map using a filter function. Only parameters for
        which the filter function returns true will remain in this map.

        :param filterFunction: any function that maps quantities to boolean values
        :return:
        """
        for key in self.__optimisableToParameterMap.keys():
            filtered = filter(filterFunction, self.__optimisableToParameterMap[key])
            self.__optimisableToParameterMap[key] = list(filtered)
