"""Class definition for the Quantity model."""

from __future__ import annotations  # necessary for type hints

import copy
from typing import Self
from collections.abc import Callable

import numpy as np

from cthree.Exceptions import IncompatibleQuantityException


class Quantity:
    r"""Represent any physical quantity used in model or pulse specification.

    For arithmetic operations just the numeric value is used.
    The value itself is stored in an optimizer friendly way as a float
    between -1 and 1. The conversion is given by
    scale * (value + 1) / 2 + offset

    Note on python's operators: equality checks `q == p` and `q != p` check
    for the values of the quantities q and p. For vector or matrix quantities,
    these check if all values are equal. If you want to be sure that two
    quantities are the same object (i.e. the same memory address), use `q is p`.
    Ordering operators like `q > p` will only work for scalar quantities and
    will raise an exception for vector or matrix quantities.

    Parameters
    ----------
    value : numpy.array(numpy.float64) | numpy.float64
        Value of the quantity
    min_value : numpy.array(numpy.float64) or numpy.float64
        Minimum this quantity is allowed to take.
        If this is null, a default interval around the value will be chosen.
    max_value : numpy.array(numpy.float64) or numpy.float64
        Maximum this quantity is allowed to take.
        If this is null, a default interval around the value will be chosen.
    unit : str
        physical unit
    name : str
        symbol or description of this quantity
    two_pi : bool
        divide by two pi for representation

    Raises
    ------
    Exception
        If the value, minimum or maximum is null, raise an Exception.

    """

    __unit: str
    __name: str
    __length: int
    __shape: tuple
    # internal representation of the value
    __value: np.ndarray
    __offset: np.ndarray
    __scale: np.ndarray
    __twoPi: bool
    __dependent: bool
    __dependencies: list
    __relation: Callable
    __dependents: list

    def __init__(
        self,
        value: np.array,
        min_value: np.ndarray,
        max_value: np.ndarray,
        unit: str = "",
        name: str = "",
        two_pi: bool = False,
    ):
        if value is None or max_value is None or min_value is None:
            raise Exception("value, minimum, and maximum must be not null")

        self.__unit = unit
        self.__name = name
        self.__scale = np.array(0)
        self.__twoPi = two_pi

        if np.shape(value) == ():
            value = np.array([value])
        else:
            value = np.array(value)

        self.__shape = value.shape
        self.__length = int(np.prod(value.shape))

        self.__offset = np.array(min_value)
        self.__scale = np.abs(np.array(max_value) - np.array(min_value))

        # if this quantity is dependent on/calculated from other quantities
        self.__dependent = False
        # all Quantities that this quantity calculates from
        self.__dependencies = list()
        # the relation function that calculated value from dependencies
        self.__relation = None

        # all Quantities that use this quantity to calculate value
        self.__dependents = list()

        self.set_value(value)

    @property
    def dependencies(self) -> list[Quantity]:
        """If dependent, get a list of parameter dependencies.

        If calculated from other quantities by a relation, this method returns
        the list of parameters that this quantity is calculated from,
        otherwise an empty list is returned.

        Returns
        -------
        List[Quantity]
            List of parameter dependencies.
        """
        if self.__dependent:
            return self.__dependencies
        return []

    @property
    def dependents(self) -> list[Quantity]:
        """Get a list of parameter dependents.

        This method returns the list of parameters that use this Quantity to
        calculate its value from. If no other Quantities calculate their value
        using this quantity, returns an empty list.

        Returns
        -------
        List[Quantity]
            List of parameter dependencies.
        """
        return self.__dependents

    @classmethod
    def relational(
        cls,
        quantities: Quantity | list[Quantity],
        relation: Callable,
        unit: str | None = None,
        name: str | None = None,
        two_pi: bool = False,
    ) -> Self:
        """Create a relationally derived parameter.

        Creates a Quantity object that represents a Quantity that is calculated
        from other quantities using the relation function.

        Parameters
        ----------
        quantities: Quantity | List[Quantity]
            The quantities from which to calculate the value of self.
        relation: Callable
            Function describing how to calculate the value of self from other
            Quantities.
        unit: str | None
            The unit of the resulting Quantity. If 'None', then the units of all
            quantities are assumed the same.
        name: str | None
            A string identifier name of the resulting Quantity. If 'None', then
            a name is generated from the names of the related Quantities.
        two_pi: bool
            Divide by two pi for representation.

        Returns
        -------
        Quantity
            The Quantity with a relation set up, which recalculates the value of
            self from all dependencies.

        Raises
        ------
            ValueError:
                If any quantities do not have the same unit and no special
                unit is specified.

        """
        quantities = quantities if isinstance(quantities, list) else [quantities]
        min_val = np.min([qty.get_min_value() for qty in quantities])
        max_val = np.max([qty.get_max_value() for qty in quantities])

        if name is None:
            name = "relation_of"
            for qty in quantities:
                name += "_" + qty.get_name()

        if unit is None:
            if not all(qty.get_unit() == quantities[0].get_unit() for qty in quantities):
                raise ValueError(
                    f"All quantities in creation on {name} " f"must have the same unit if no unit is specified."
                )
            unit = quantities[0].get_unit()

        qty = cls(
            value=min_val,
            min_value=min_val,
            max_value=max_val,
            unit=unit,
            name=name,
            two_pi=two_pi,
        )
        qty.add_relation(quantities, relation)
        return qty

    @classmethod
    def relational_copy(cls, quantity: Quantity) -> Self:
        """Create a Quantity object that is a one to one copy of a Quantity.

        If the quantity is updated, so is this relational copy.

        Parameters
        ----------
        quantity: Quantity | List[Quantity]
            The quantities from which the relational copy should be created.

        Returns
        -------
        Quantity
            The Quantity with a relation set up, which recalculates the value of
            self from all dependencies.
        """
        qty = cls(
            value=quantity.get_value(),
            min_value=quantity.get_min_value(),
            max_value=quantity.get_max_value(),
            unit=quantity.get_unit(),
            name=quantity.get_name(),
            two_pi=quantity.__twoPi,
        )
        qty.add_relation(quantity, lambda x: x)
        return qty

    @property
    def dependent(self):
        """The dependency status of the quantity.

        if True:
            The value of this quantity is calculated from other quantities
        if False:
            The value of this quantity is independent of any other quantity
        """
        return self.__dependent

    def add_relation(
        self,
        other: Quantity | list[Quantity],
        relation: Callable,
        checkUnits: bool = True,
    ) -> None:
        """Add a relation of self to one or more other quantities.

        Parameters
        ----------
        other: Quantity | List[Quantity]
            The quantities from which to calculate the value of self.
        relation: Callable
            Function describing how to calculate the value of self from other
            Quantities.
        checkUnits: bool
            If False, the check for equal units is not performed and unequal
            units are allowed.

        """
        other = other if isinstance(other, list) else [other]

        if not all(qty.get_unit() == self.get_unit() for qty in other) and checkUnits:
            raise ValueError(
                "Not all Quantities in the relation have the same units. "
                "This may lead to unintentional physical errors. "
                "Set 'checkUnits=False' if this behavior is wanted."
            )

        self.__dependent = True
        self.__dependencies = other
        self.__relation = relation
        self.update()

        for qty in self.__dependencies:
            qty.__dependents.append(self)

    def update(self):
        """Update value of the parameter.

        Update function that is called if a value that this quantity
        is dependent on is changed.

        """
        self.__set_value(self.__relation(*[qty.get_value() for qty in self.__dependencies]))

    def get_value(self) -> np.array:
        """Get value of the parameter."""
        return self.__scale * (self.__value + 1) / 2 + self.__offset

    def get_reduced_value(self) -> np.ndarray:
        """Return the value in the reduced representation.

        Returns
        -------
        numpy.ndarray
            Value from the reduced representation.

        """
        return np.reshape(self.__value, (-1, 1))

    def set_value(self, value) -> None:
        """Set the value of this quantity.

        Value needs to be within the range of 'min_value' and 'max_value'.

        Parameters
        ----------
        value
            Input value to be used for setting.

        Raises
        ------
        ValueError
            If the value is not within the range of 'min_value' and 'max_value'.

        """
        if self.__dependent:
            raise ValueError(
                "Cannot set value on dependent quantities, \
                    as it is calculated from other quantities."
            )

        self.__set_value(value)

    def __set_value(self, value) -> None:
        """Set value for the parameter."""
        if isinstance(value, np.ndarray):
            val = value.astype(np.float64)
        else:
            val = np.array(value).astype(np.float64)
        tmp = 2 * (np.reshape(val, self.__shape) - self.__offset) / self.__scale - 1

        if np.any(np.abs(tmp) > 1.0):
            print("Error: ", val, self.get_min_value(), self.get_max_value())
            raise ValueError(
                f"Value \
                    {self.__to_string(val)}{self.__unit} \
                        out of bounds for quantity with "
                f"min_val: \
                    {self.__to_string(self.get_min_value())}{self.__unit} and "
                f"max_val: {self.__to_string(self.get_max_value())}{self.__unit}",
            )
        self.__value = tmp

        # update all Quantities that depend on self
        for qty in self.__dependents:
            qty.update()

    def set_reduced_value(self, value) -> None:
        """Set reduced value limit for parameter."""
        if np.shape(value) == ():
            value = np.array([value])
        self.__value = value

    def get_min_value(self) -> np.ndarray:
        """Get minimum value of parameter."""
        return self.__offset

    def get_max_value(self) -> np.ndarray:
        """Get maximum value of parameter."""
        return self.__scale + self.__offset

    def get_scale(self) -> np.ndarray:
        """Get scale of parameter."""
        return self.__scale

    def get_length(self) -> int:
        """Get length of parameter."""
        return self.__length

    def set_limits(self, min_value, max_value) -> None:
        """Set the allowed minimum and maximum of this quantity.

        Parameters
        ----------
        min_value : int
            Input value for setting the minimum limit.
        max_value : int
            Input value for setting the maximum limit.

        """
        oldValue = self.get_value()
        self.__offset = np.array(min_value)
        self.__scale = np.abs(np.array(max_value) - np.array(min_value))
        # the value is based on offset and scale and needs to be updated
        self.__set_value(oldValue)

    def set_value_and_limits(self, value, min_value, max_value) -> None:
        """
        This can be used to set the value and the limits to new values at the same time. This function does not raise
        an exception if the new value is outside of the old limits.
        """
        self.__offset = np.array(min_value)
        self.__scale = np.abs(np.array(max_value) - np.array(min_value))
        self.__set_value(value)

    def get_name(self) -> str:
        """Return the symbol or description or this quantity.

        Note that this does not have to be unique.
        For uniquely identifying a quantity, use getUUID.

        Returns
        -------
        str
            Value of the name attribute.

        """
        return self.__name

    def set_name(self, name: str) -> None:
        """Assigns a new name to this quantity."""
        self.__name = name

    def get_unit(self) -> str:
        """Get unit of measurement from paramter."""
        return self.__unit

    def is_scalar(self) -> bool:
        """Check if parameter is scalar."""
        return self.__length == 1

    def is_vector(self) -> bool:
        """Check if parameter is vector."""
        return self.__length > 1 and len(self.__shape) == 1

    # Python specific functions
    def __add__(self, other) -> Quantity:
        """Magic method for addition by operand."""
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() + other)
        return out_val

    def __radd__(self, other) -> Quantity:
        """Magic method for addition by right-hand operand."""
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() + other)
        return out_val

    def __sub__(self, other) -> Quantity:
        """Magic method for subtraction by operand."""
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() - other)
        return out_val

    def __rsub__(self, other) -> Quantity:
        """Magic method for subtraction by right-hand operand."""
        out_val = copy.deepcopy(self)
        out_val.set_value(other - self.get_value())
        return out_val

    def __mul__(self, other) -> Quantity:
        """Magic method for multiplication by operand."""
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() * other)
        return out_val

    def __rmul__(self, other) -> Quantity:
        """Magic method for multiplication by right-hand operand."""
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() * other)
        return out_val

    def __pow__(self, other) -> Quantity:
        """Magic method for exponentiation by operand."""
        out_val = copy.deepcopy(self)
        out_val.set_value(np.float_power(self.get_value(), other))
        return out_val

    def __rpow__(self, other) -> Quantity:
        """Magic method for exponentiation by right-hand operand."""
        out_val = copy.deepcopy(self)
        out_val.set_value(np.float_power(other, self.get_value()))
        return out_val

    def __truediv__(self, other) -> Quantity:
        """Magic method for division by operand."""
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() / other)
        return out_val

    def __rtruediv__(self, other) -> Quantity:
        """Magic method for division by right-hand operand."""
        out_val = copy.deepcopy(self)
        out_val.set_value(other / self.get_value())
        return out_val

    def __mod__(self, other) -> Quantity:
        """Magic method for representation of the modulus operation."""
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() % other)
        return out_val

    def __lt__(self, other) -> bool:
        """Magic method for representation of less-than operation.

        Raises
        ------
        cthree.Exceptions.IncompatibleQuantityException
            If the parameter is incompatible for this operation.

        Returns
        -------
        bool
            True if self's value is less than other paramter's value.
            Note: Because mypy doesn't understand what the type of
            'self.getValue' and 'other.getValue' is, the return
            type might have to be written as 'Any'.

        """
        if not self.is_scalar():
            raise IncompatibleQuantityException("Ordering operators are only usable with scalar quantities")
        return self.get_value() < other.get_value()

    def __le__(self, other) -> bool:
        """Magic method for representation of less-equal operation.

        Raises
        ------
        cthree.Exceptions.IncompatibleQuantityException
            If the parameter is incompatible for this operation.

        Returns
        -------
        bool
            True if self's value is less than or equal to the operand's
            value.
            Note: Because mypy doesn't understand what the type of
            'self.getValue' and 'other.getValue' is, the return
            type might have to be written as 'Any'.

        """
        if not self.is_scalar():
            raise IncompatibleQuantityException("Ordering operators are only usable with scalar quantities")
        return self.get_value() <= other

    def __eq__(self, other) -> bool:
        """Magic method for representation of equality operation."""
        if self.__shape != other.__shape:
            return False
        return all(self.get_value() == other)

    def __ne__(self, other) -> bool:
        """Magic method for representation of not-equal operation."""
        if self.__shape != other.__shape:
            return True
        return any(self.get_value() != other)

    def __ge__(self, other) -> bool:
        """Magic method for representation of greater-equal operation.

        Raises
        ------
        cthree.Exceptions.IncompatibleQuantityException
            If the parameter is incompatible for this operation.

        Returns
        -------
        bool
            True if self's value is greater than or equal to the operand's
            value.
            Note: Because mypy doesn't understand what the type of
            'self.getValue' and 'other.getValue' is, the return
            type might have to be written as 'Any'.

        """
        if not self.is_scalar():
            raise IncompatibleQuantityException("Ordering operators are only usable with scalar quantities")
        return self.get_value() >= other

    def __gt__(self, other) -> bool:
        """Magic method for representation of greater-than operation.

        Raises
        ------
        cthree.Exceptions.IncompatibleQuantityException
            If the parameter is incompatible for this operation.

        Returns
        -------
        bool
            True if self's value is greater than the operand's value.
            Note: Because mypy doesn't understand what the type of
            'self.getValue' and 'other.getValue' is, the return
            type might have to be written as 'Any'.

        """
        if not self.is_scalar():
            raise IncompatibleQuantityException("Ordering operators are only usable with scalar quantities")
        return self.get_value() > other

    def __array__(self):
        """Magic method for representation into array."""
        return np.array(self.get_value())

    def __len__(self):
        """Magic method for calculation of length."""
        return self.__length

    def __getitem__(self, key):
        """Magic method for selection of item.

        Parameters
        ----------
        key : int
            Index of object for retrieval.

        """
        if self.__length == 1 and key == 0:
            return self.get_value()
        return self.get_value().__getitem__(key)

    def __abs__(self):
        """Magic method for absolute value calculation."""
        return abs(self.get_value())

    def __float__(self):
        """Magic method for float coversion.

        Raises
        ------
        NotImplementedError
            If the length of the parameter is greater than 1.

        """
        if self.__length > 1:
            raise NotImplementedError
        return float(np.squeeze(self.get_value()))

    def __repr__(self):
        """Magic method for human readable representation."""
        return self.__str__()

    def __str__(self):
        """Human readable representation of the parameters set to optimise."""
        return self.__to_string(self.get_value())

    def __to_string(self, val):
        """Represent parameter as custom defined string value."""
        ret = ""
        for entry in np.nditer(val):
            if self.__unit != "":
                if self.__twoPi:
                    ret += self.__make_human_readable(entry / np.pi / 2) + self.__unit + " x 2pi "
                else:
                    ret += self.__make_human_readable(entry) + self.__unit + " "
            else:
                if self.__twoPi:
                    ret += self.__make_human_readable(entry / np.pi / 2, use_prefix=False) + " x 2pi "
                else:
                    ret += self.__make_human_readable(entry, use_prefix=False) + " "
        if self.__name:
            ret = self.__name + ": " + ret
        return ret

    @staticmethod
    def __make_human_readable(val, use_prefix: bool = True) -> str:
        """Convert to human readable string in engineering notation.

        Parameters
        ----------
        use_prefix : bool, default=True
            Adds a prefix string derived from '__engineeringNumber'
            to the final format string.

        Returns
        -------
        str
            Human readable formatted string for value.

        """
        if use_prefix:
            num, prefix = Quantity.__engineering_number(val)
            formatted_string = f"{num:.3g} " + prefix
        else:
            formatted_string = f"{val:.3g} "
        return formatted_string

    # Internal utility functions
    @staticmethod
    def __engineering_number(val: float) -> tuple[float, str]:
        """Convert number to engineering notation.

        Returns number and prefix.

        Parameters
        ----------
        val : float
            Input number to be converted to engineering notation.

        Returns
        -------
        Tuple[float, str]
            Engineering notation composite made of the number and the prefix.

        """
        if np.isnan(val):
            return np.nan, "NaN"

        sign = 1.0
        if val == 0:
            return 0.0, ""
        if val < 0.0:
            val = -val
            sign = -1.0
        tmp = np.log10(val)
        idx = int(tmp // 3)

        if tmp < 0:
            units = ["m", "µ", "n", "p", "f", "a", "z"]
            if np.abs(idx) > len(units):
                return val * (10 ** (3 * len(units))), units[-1]
            prefix = units[-(idx + 1)]
        else:
            units = ["", "K", "M", "G", "T", "P", "E", "Z"]
            if np.abs(idx) > len(units) - 1:
                return val * (10 ** (-3 * (len(units) - 1))), units[-1]
            prefix = units[idx]

        return sign * (10 ** (tmp % 3)), prefix

    def to_dict(self) -> dict:
        """
        Creates a dictionary representation of this quantity that can be stored. The returned dict is compatible with
        the fromDict function, i.e. the quantity can be fully restored including its bounds, name, unit, etc. Higher
        dimensional quantities (tensors) will be flattened into a list but their proper shape is stored as well.
        """
        if self.dependent:
            raise UserWarning("Saving of dependent quantities is not supported yet")

        return {
            "unit": self.__unit,
            "shape": self.__shape,
            "twoPi": self.__twoPi,
            "value": self.get_value().flatten().tolist(),
            "min": self.get_min_value().tolist(),
            "max": self.get_max_value().tolist(),
        }

    def from_dict(self, data: dict) -> None:
        """
        Loads the quantity from a dictionary. The dictionary must have the same form as the one created by the toDict
        function. All properties of this quantity (value, name, etc.) will be overwritten.
        """
        self.__unit = data["unit"]
        self.__shape = data["shape"]
        self.__length = int(np.prod(self.__shape))
        self.__twoPi = data["twoPi"]

        # The value and limits need to be set at the same time so that the new value is not out of range
        value = np.array(data["value"]).reshape(self.__shape)
        minVal = np.array(data["min"]).reshape(self.__shape)
        maxVal = np.array(data["max"]).reshape(self.__shape)
        self.set_value_and_limits(value, minVal, maxVal)
