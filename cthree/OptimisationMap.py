"""Class definition for the Optimisable Map model."""

from collections.abc import Callable

from cthree.Exceptions import SerialisationException
from cthree.Optimisable import Optimisable
from cthree.Quantity import Quantity


class OptimisationMap:
    """Optimisation parameter map utility class.

    Utility class that collects all parameters that shall be considered during
    optimisation and associates them with the corresponding Optimisable
    interface. With this class, Quantities can be traced back to the
    Optimisable to which they belong. Before optimisation, an instance of this
    class needs to be filled and passed to the optimiser.

    """

    __optimisableToParameterMap: dict[Optimisable, list[Quantity]]

    def __init__(self):
        self.__optimisableToParameterMap = {}

    def __repr__(self):
        """Magic method for human readable representation."""
        return self.__str__()

    def __str__(self):
        """Human readable representation of the parameters set to optimise."""
        om_str = ""
        for key, val in self.__optimisableToParameterMap.items():
            om_str += f"==== {key} ====\n"
            om_str += str(val)
            om_str += "\n\n"
        return om_str

    def add(
        self,
        optimisable: Optimisable,
        optimisableQuantities: list[Quantity] = None,
    ):
        """Add an optimisable object and a list of its quantities to the map.

        The list contains all parameters of the optimisable object that shall
        be considered during the optimisation. If the list is empty, all
        parameters of the class will be used. If the object was already added,
        the list of quantities will be overwritten.

        Parameters
        ----------
        optimisable : cthree.Optimisable
            Input Optimisable object for adding to the map.
        optimisableQuantities : List[cthree.Quantity], optional
            List of all parameters of the optimisable object considered for
            optimisation.

        """
        params = optimisableQuantities or optimisable.getParameters()
        self.__optimisableToParameterMap[optimisable] = params
        if len(self.__optimisableToParameterMap[optimisable]) < 1:
            self.__optimisableToParameterMap.pop(optimisable)

    def remove(self, optimisable: Optimisable):
        """Remove the given parameter from the sytem.

        Parameters
        ----------
        optimisable : cthree.Optimisable.Optimisable
            Parameter to be removed.

        """
        try:
            self.__optimisableToParameterMap.pop(optimisable)
        # removed the bare except catch.
        except Exception as e:
            raise Exception(e)

    def getOptimisables(self) -> set[Optimisable]:
        """Return all optimisable objects that were added to this map.

        Returns
        -------
        Set[cthree.Optimisable]
            Set of all optimisable objects from the map.

        """
        return set(self.__optimisableToParameterMap.keys())

    def getParameters(self, optimisable: Optimisable) -> list[Quantity] | None:
        """Return all quantities associated with the given parameter.

        Parameters
        ----------
        optimisable : cthree.Optimisable
            Input optimisable object.

        Returns
        -------
        List[cthree.Quantity] | None
            List of parameters or None (if the optimisable has not been
            added yet).

        """
        return self.__optimisableToParameterMap[optimisable]

    def getAllParameters(self) -> list[Quantity]:
        """Return all parameters that were added to the system map.

        Returns
        -------
        List[cthree.Quantity]
            All parameters that were added to the map.

        """
        quantities = []
        for params in self.__optimisableToParameterMap.values():
            quantities.extend(params)
        return quantities

    def registerParamsWithOptimisables(self) -> None:
        """Register optimisable parameters with the system.

        Utility function that synchronises the list of parameters with
        each optimisable class. This needs to be called by the optimiser
        before gradient based optimisation to tell the layers which gradients
        to compute.

        """
        for optimisable, params in self.__optimisableToParameterMap.items():
            optimisable.setOptimisableParameters(params)

    def filterParameters(self, filterFunction: Callable) -> None:
        """Filter parameters using filter function.

        Updates the list of parameters for all Optimisables in this map using
        a filter function. Only parameters for which the filter function returns
        true will remain in this map.

        Parameters
        ----------
        filterFunction : Callable
            Filter function that maps quantities to boolean values.

        """
        for key in self.__optimisableToParameterMap.keys():
            filtered = filter(
                filterFunction, self.__optimisableToParameterMap[key]
            )
            self.__optimisableToParameterMap[key] = list(filtered)
        self.__optimisableToParameterMap = dict(
            (k, v)
            for k, v in self.__optimisableToParameterMap.items()
            if len(v) > 0
        )

    def filterByName(self, name: str):
        """Filter parameters by name of parameter.

        Parameters
        ----------
        name : str
            Name of parameter to be filtered with.

        """
        return self.filterParameters(
            lambda quantity: quantity.getName() == name
        )

    def toDict(self) -> dict:
        data = dict()
        for optimisable, quantities in self.__optimisableToParameterMap.items():
            if len((optimisable.name or '').strip()) == 0:
                raise SerialisationException(
                    'Optimisable does not have a name. Serialisation is only possible if the name of the optimisable is unique.')
            for q in quantities:
                if len((q.getName() or '').strip()) == 0:
                    raise SerialisationException(
                        'Quantity does not have a name. Serialisation is only possible if the name of a quantity is unique within the optimisable.')
            data[optimisable.name] = [q.toDict() for q in quantities]
        return data
