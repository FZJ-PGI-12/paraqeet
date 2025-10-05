"""Class definition for the Optimisable Map model."""

from collections.abc import Callable

from paraqeet.exceptions import SerialisationException
from paraqeet.optimisable import Optimisable
from paraqeet.quantity import Quantity


class OptimisationMap:
    """Optimisation parameter map utility class.

    Utility class that collects all parameters that shall be considered during
    optimisation and associates them with the corresponding Optimisable
    interface. With this class, Quantities can be traced back to the
    Optimisable to which they belong. Before optimisation, an instance of this
    class needs to be filled and passed to the optimiser.

    """

    __optimisable_to_parameter_map: dict[Optimisable, list[Quantity]]

    def __init__(self):
        self.__optimisable_to_parameter_map = {}

    def __repr__(self):
        """Magic method for human readable representation."""
        return self.__str__()

    def __str__(self):
        """Human readable representation of the parameters set to optimise."""
        om_str = ""
        for key, val in self.__optimisable_to_parameter_map.items():
            om_str += f"==== {key} ====\n"
            om_str += str(val)
            om_str += "\n\n"
        return om_str

    def add(
        self,
        optimisable: Optimisable,
        optimisable_quantities: Quantity | list[Quantity] | None = None,
    ):
        """Add an optimisable object and a list of its quantities to the map.

        The list contains all parameters of the optimisable object that shall
        be considered during the optimisation. If the list is empty, all
        parameters of the class will be used. If the object was already added,
        the list of quantities will be overwritten.

        Parameters
        ----------
        optimisable: Optimisable
            Input Optimisable object for adding to the map.
        optimisable_quantities: Quantity | list[Quantity] | None = None
            List of all parameters of the optimisable object considered for
            optimisation.

        """
        if optimisable_quantities is not None:
            params: list[Quantity] = (
                [optimisable_quantities] if isinstance(optimisable_quantities, Quantity) else optimisable_quantities
            )
        else:
            params = optimisable.get_parameters()
        self.__optimisable_to_parameter_map[optimisable] = params
        if len(self.__optimisable_to_parameter_map[optimisable]) < 1:
            self.__optimisable_to_parameter_map.pop(optimisable)

    def append(
        self,
        optimisable: Optimisable,
        optimisable_quantities: Quantity | list[Quantity] | None = None,
    ):
        """Append an optimisable object and a list of its quantities to the map.

        This method is similar to the `add` method, but instead of overwritting the
        existing entries, this appends the specified list of quantities to the
        already existing quantities.

        Parameters
        ----------
        optimisable: Optimisable
            Input Optimisable object for adding to the map.
        optimisable_quantities: Quantity | list[Quantity] | None = None
            List of all parameters of the optimisable object considered for
            optimisation.

        """
        if optimisable_quantities is not None:
            params: list[Quantity] = (
                [optimisable_quantities] if isinstance(optimisable_quantities, Quantity) else optimisable_quantities
            )
        else:
            params = optimisable.get_parameters()

        if optimisable in self.__optimisable_to_parameter_map:
            self.__optimisable_to_parameter_map[optimisable].extend(params)
        else:
            self.__optimisable_to_parameter_map[optimisable] = params
        if len(self.__optimisable_to_parameter_map[optimisable]) < 1:
            self.__optimisable_to_parameter_map.pop(optimisable)

    def remove(self, optimisable: Optimisable, params: Quantity | list[Quantity] | None = None):
        """Remove the given optimisable or parameter(s) from the optimisation map.

        If params is None, it removes the optimisable from the optimisation map.
        Else it only removes the specific parameter from the optimisation map.

        Parameters
        ----------
        optimisable: Optimisable
            Optimisable to be removed.
        params: Quantity | list[Quanitity] | None.
            Parameter(s) to be removed from the optimisation map. If None removes the optimisable.

        """
        try:
            if params is None:
                self.__optimisable_to_parameter_map.pop(optimisable)
            else:
                parameters_list: list[Quantity] = params if isinstance(params, list) else [params]
                self.filter_parameters(lambda quantity: quantity not in parameters_list)

        # removed the bare except catch.
        except Exception as e:
            raise Exception(e)

    def replace(
        self,
        optimisable: Optimisable,
        old_parameters: Quantity | list[Quantity],
        new_parameters: Quantity | list[Quantity],
    ) -> None:
        """Perform an in-place substitution of the old and new parameters.
        This helps to keeps the ordering of parameters the same while replacing parameters.
        """
        old_parameters_list = old_parameters if isinstance(old_parameters, list) else [old_parameters]
        new_parameters_list = new_parameters if isinstance(new_parameters, list) else [new_parameters]

        for old_param, new_param in zip(old_parameters_list, new_parameters_list):
            for i, param in enumerate(self.__optimisable_to_parameter_map[optimisable]):
                if id(param) == id(old_param):
                    self.__optimisable_to_parameter_map[optimisable][i] = new_param

    def get_optimisables(self) -> set[Optimisable]:
        """Return all optimisable objects that were added to this map.

        Returns
        -------
        set[Optimisable]
            Set of all optimisable objects from the map.

        """
        return set(self.__optimisable_to_parameter_map.keys())

    def get_parameters(self, optimisable: Optimisable) -> list[Quantity] | None:
        """Return all quantities associated with the given parameter.

        Parameters
        ----------
        optimisable: Optimisable
            Input optimisable object.

        Returns
        -------
        list[Quantity] | None
            List of parameters or None (if the optimisable has not been
            added yet).

        """
        return self.__optimisable_to_parameter_map[optimisable]

    def get_all_parameters(self) -> list[Quantity]:
        """Return all parameters that were added to the system map.

        Returns
        -------
        List[Quantity]
            All parameters that were added to the map.

        """
        quantities = []
        for params in self.__optimisable_to_parameter_map.values():
            quantities.extend(params)
        return quantities

    def register_params_with_optimisables(self) -> None:
        """Register optimisable parameters with the system.

        Utility function that synchronises the list of parameters with
        each optimisable class. This needs to be called by the optimiser
        before gradient based optimisation to tell the layers which gradients
        to compute.

        """
        for optimisable, params in self.__optimisable_to_parameter_map.items():
            optimisable.set_optimisable_parameters(params)
            optimisable.set_all_optimisable_parameters(self.get_all_parameters())

    def filter_parameters(self, filter_function: Callable) -> None:
        """Filter parameters using filter function.

        Updates the list of parameters for all Optimisables in this map using
        a filter function. Only parameters for which the filter function returns
        true will remain in this map.

        Parameters
        ----------
        filter_function : Callable
            Filter function that maps quantities to boolean values.

        """
        for key in self.__optimisable_to_parameter_map.keys():
            filtered = filter(filter_function, self.__optimisable_to_parameter_map[key])
            self.__optimisable_to_parameter_map[key] = list(filtered)
        self.__optimisable_to_parameter_map = dict(
            (k, v) for k, v in self.__optimisable_to_parameter_map.items() if len(v) > 0
        )

    def filter_by_name(self, name: str):
        """Filter parameters by name of parameter.

        Parameters
        ----------
        name : str
            Name of parameter to be filtered with.

        """
        return self.filter_parameters(lambda quantity: quantity.get_name() == name)

    def to_dict(self) -> dict:
        """Creates a dictionary that contains the values of all quantities that are being optimised, sorted by the
        Optimisable instances to which they belong. The returned dictionary is meant for export using the serialisation
        package. It uses the names of Optimisables and Quantities and assumes that those are unique and not None. The
        format of the dict will be

        "optimisable name": {
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
            all optimised quantities in an exportable format

        Raises
        ------
        SerialisationException
            If the name of any Optimisable or Quantity is None or not unique.
        """
        data = dict()
        for optimisable, quantities in self.__optimisable_to_parameter_map.items():
            # Check that the optimisable's name is valid
            if len((optimisable.name or "").strip()) == 0 or optimisable.name in data:
                raise SerialisationException("Optimisable does not have a name or the name is not unique.")

            # Check that the quantities' names are valid
            quantity_names = [(q.get_name() or "").strip() for q in quantities]
            non_empty_quantity_names = list(filter(lambda name: len(name) > 0, quantity_names))
            if len(quantities) != len(set(non_empty_quantity_names)):
                raise SerialisationException(
                    f"Quantities in {optimisable.name} have empty or non-unique names within the optimisable."
                )

            data[optimisable.name] = {q.get_name(): q.to_dict() for q in quantities}
        return data

    def from_dict(self, data: dict) -> None:
        """
        Restores the values of all optimised quantities that are in the dictionary. The format of the dictionary needs
        to be in the same format as generated by the toDict function.

        Parameters
        ----------
        data: dict
            All quantities that should be restored.

        Raises
        ------
        SerialisationException
            If the dict contains an Optimisable or a Quantity that does not exist in this optimisation map.
        """
        optimisables_for_name = {opt.name: opt for opt in self.__optimisable_to_parameter_map.keys()}
        for optimisable_name, values in data.items():
            if optimisable_name not in optimisables_for_name:
                raise SerialisationException(
                    f'An optimisable with the name "{optimisable_name}" does not exist in the optimisation map.'
                )
            optimisable = optimisables_for_name[optimisable_name]

            quantities_for_name = {q.get_name(): q for q in optimisable.get_parameters()}
            for quantity_name, quantity_values in values.items():
                if quantity_name not in quantities_for_name:
                    raise SerialisationException(f'Quantity "{quantity_name}" does not exist in {optimisable_name}.')
                quantities_for_name[quantity_name].from_dict(quantity_values)
