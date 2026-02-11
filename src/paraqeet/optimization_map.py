"""Class definition for the Optimizable Map model."""

from collections.abc import Callable

from paraqeet.exceptions import SerializationException
from paraqeet.optimizable import Optimizable
from paraqeet.quantity import Quantity


class OptimizationMap:
    """Optimization parameter map utility class.

    Utility class that collects all parameters that shall be considered during
    optimization and associates them with the corresponding Optimizable
    interface. With this class, Quantities can be traced back to the
    Optimizable to which they belong. Before optimization, an instance of this
    class needs to be filled and passed to the optimizer.

    """

    _optimizable_to_parameter_map: dict[Optimizable, list[Quantity]]

    def __init__(self):
        self._optimizable_to_parameter_map = {}

    def __repr__(self):
        """Magic method for human readable representation."""
        return self.__str__()

    def __str__(self):
        """Human readable representation of the parameters set to optimize."""
        om_str = ""
        for key, val in self._optimizable_to_parameter_map.items():
            om_str += f"==== {key} ====\n"
            om_str += str(val)
            om_str += "\n\n"
        return om_str

    def add(
        self,
        optimizable: Optimizable,
        optimizable_quantities: Quantity | list[Quantity] | None = None,
    ):
        """Add an optimizable object and a list of its quantities to the map.

        The list contains all parameters of the optimizable object that shall
        be considered during the optimization. If the list is empty, all
        parameters of the class will be used. If the object was already added,
        the list of quantities will be overwritten.

        Parameters
        ----------
        optimizable: Optimizable
            Input Optimizable object for adding to the map.
        optimizable_quantities: Quantity | list[Quantity] | None = None
            List of all parameters of the optimizable object considered for
            optimization.

        """
        if optimizable_quantities is not None:
            params: list[Quantity] = (
                [optimizable_quantities] if isinstance(optimizable_quantities, Quantity) else optimizable_quantities
            )
        else:
            params = optimizable.get_parameters()
        self._optimizable_to_parameter_map[optimizable] = params
        if len(self._optimizable_to_parameter_map[optimizable]) < 1:
            self._optimizable_to_parameter_map.pop(optimizable)

    def append(
        self,
        optimizable: Optimizable,
        optimizable_quantities: Quantity | list[Quantity] | None = None,
    ):
        """Append an optimizable object and a list of its quantities to the map.

        This method is similar to the `add` method, but instead of overwriting the
        existing entries, this appends the specified list of quantities to the
        already existing quantities.

        Parameters
        ----------
        optimizable: optimizable
            Input optimizable object for adding to the map.
        optimizable_quantities: Quantity | list[Quantity] | None = None
            List of all parameters of the optimizable object considered for
            optimization.

        """
        if optimizable_quantities is not None:
            params: list[Quantity] = (
                [optimizable_quantities] if isinstance(optimizable_quantities, Quantity) else optimizable_quantities
            )
        else:
            params = optimizable.get_parameters()

        if optimizable in self._optimizable_to_parameter_map:
            self._optimizable_to_parameter_map[optimizable].extend(params)
        else:
            self._optimizable_to_parameter_map[optimizable] = params
        if len(self._optimizable_to_parameter_map[optimizable]) < 1:
            self._optimizable_to_parameter_map.pop(optimizable)

    def remove(self, optimizable: Optimizable, params: Quantity | list[Quantity] | None = None):
        """Remove the given optimizable or parameter(s) from the optimization map.

        If params is None, it removes the optimizable from the optimization map.
        Else it only removes the specific parameter from the optimization map.

        Parameters
        ----------
        optimizable: optimizable
            optimizable to be removed.
        params: Quantity | list[Quantity] | None.
            Parameter(s) to be removed from the optimization map. If None removes the optimizable.

        """
        try:
            if params is None:
                self._optimizable_to_parameter_map.pop(optimizable)
            else:
                parameters_list: list[Quantity] = params if isinstance(params, list) else [params]
                self.filter_parameters(lambda quantity: quantity not in parameters_list)

        # removed the bare except catch.
        except Exception as e:
            raise Exception(e)

    def replace(
        self,
        optimizable: Optimizable,
        old_parameters: Quantity | list[Quantity],
        new_parameters: Quantity | list[Quantity],
    ) -> None:
        """Perform an in-place substitution of the old and new parameters.
        This helps to keeps the ordering of parameters the same while replacing parameters.
        """
        old_parameters_list = old_parameters if isinstance(old_parameters, list) else [old_parameters]
        new_parameters_list = new_parameters if isinstance(new_parameters, list) else [new_parameters]

        for old_param, new_param in zip(old_parameters_list, new_parameters_list):
            for i, param in enumerate(self._optimizable_to_parameter_map[optimizable]):
                if id(param) == id(old_param):
                    self._optimizable_to_parameter_map[optimizable][i] = new_param

    def get_optimizables(self) -> set[Optimizable]:
        """Return all optimizable objects that were added to this map.

        Returns
        -------
        set[Optimizable]
            Set of all optimizable objects from the map.

        """
        return set(self._optimizable_to_parameter_map.keys())

    def get_parameters(self, optimizable: Optimizable) -> list[Quantity] | None:
        """Return all quantities associated with the given parameter.

        Parameters
        ----------
        optimizable: Optimizable
            Input optimizable object.

        Returns
        -------
        list[Quantity] | None
            List of parameters or None (if the optimizable has not been
            added yet).

        """
        return self._optimizable_to_parameter_map[optimizable]

    def get_all_parameters(self) -> list[Quantity]:
        """Return all parameters that were added to the system map.

        Returns
        -------
        List[Quantity]
            All parameters that were added to the map.

        """
        quantities = []
        for params in self._optimizable_to_parameter_map.values():
            quantities.extend(params)
        return quantities

    def register_params_with_optimizables(self) -> None:
        """Register optimizable parameters with the system.

        Utility function that synchronises the list of parameters with
        each optimizable class. This needs to be called by the optimizer
        before gradient based optimization to tell the layers which gradients
        to compute.

        """
        for optimizable, params in self._optimizable_to_parameter_map.items():
            optimizable.set_optimizable_parameters(params)
            optimizable.set_all_optimizable_parameters(self.get_all_parameters())

    def filter_parameters(self, filter_function: Callable) -> None:
        """Filter parameters using filter function.

        Updates the list of parameters for all Optimizables in this map using
        a filter function. Only parameters for which the filter function returns
        true will remain in this map.

        Parameters
        ----------
        filter_function : Callable
            Filter function that maps quantities to boolean values.

        """
        for key in self._optimizable_to_parameter_map.keys():
            filtered = filter(filter_function, self._optimizable_to_parameter_map[key])
            self._optimizable_to_parameter_map[key] = list(filtered)
        self._optimizable_to_parameter_map = dict(
            (k, v) for k, v in self._optimizable_to_parameter_map.items() if len(v) > 0
        )

    def filter_by_name(self, name: str):
        """Filter parameters by name of parameter.

        Parameters
        ----------
        name : str
            Name of parameter to be filtered with.

        """
        return self.filter_parameters(lambda quantity: quantity.get_name() == name)

    def remove_by_name(self, name: str):
        """Remove parameters by name of parameter.

        Parameters
        ----------
        name : str
            Name of parameter to be filtered with.

        """
        return self.filter_parameters(lambda quantity: name not in quantity.get_name())

    def to_dict(self) -> dict:
        """Creates a dictionary that contains the values of all quantities that are being optimized, sorted by the
        Optimizable instances to which they belong. The returned dictionary is meant for export using the serialisation
        package. It uses the names of Optimizables and Quantities and assumes that those are unique and not None. The
        format of the dict will be

        "optimizable name": {
          "quantity name": {
            "unit": string,
            "shape": tuple[int, ...],
            "twoPi": bool,
            "value': Array,
            "min": Array,
            "max": Array,
          }
        }

        where the innermost part is generated by Quantity's toDict function.


        Returns
        -------
        dict
            all optimized quantities in an exportable format

        Raises
        ------
        SerialisationException
            If the name of any Optimizable or Quantity is None or not unique.

        """
        data = dict()
        for optimizable, quantities in self._optimizable_to_parameter_map.items():
            # Check that the optimizable's name is valid
            if len((optimizable.name or "").strip()) == 0 or optimizable.name in data:
                raise SerializationException("Optimizable does not have a name or the name is not unique.")

            # Check that the quantities' names are valid
            quantity_names = [(q.get_name() or "").strip() for q in quantities]
            non_empty_quantity_names = list(filter(lambda name: len(name) > 0, quantity_names))
            if len(quantities) != len(set(non_empty_quantity_names)):
                raise SerializationException(
                    f"Quantities in {optimizable.name} have empty or non-unique names within the optimizable."
                )

            data[optimizable.name] = {q.get_name(): q.to_dict() for q in quantities}
        return data

    def from_dict(self, data: dict) -> None:
        """
        Restores the values of all optimized quantities that are in the dictionary. The format of the dictionary needs
        to be in the same format as generated by the toDict function.

        Parameters
        ----------
        data: dict
            All quantities that should be restored.

        Raises
        ------
        SerialisationException
            If the dict contains an Optimizable or a Quantity that does not exist in this optimization map.
        """
        optimizables_for_name = {opt.name: opt for opt in self._optimizable_to_parameter_map.keys()}
        for optimizable_name, values in data.items():
            if optimizable_name not in optimizables_for_name:
                raise SerializationException(
                    f'An optimizable with the name "{optimizable_name}" does not exist in the optimization map.'
                )
            optimizable = optimizables_for_name[optimizable_name]

            quantities_for_name = {q.get_name(): q for q in optimizable.get_parameters()}
            for quantity_name, quantity_values in values.items():
                if quantity_name not in quantities_for_name:
                    raise SerializationException(f'Quantity "{quantity_name}" does not exist in {optimizable_name}.')
                quantities_for_name[quantity_name].from_dict(quantity_values)
