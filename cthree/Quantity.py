from __future__ import annotations  # necessary for type hints

import copy
from typing import Tuple, List, Callable, Self

import numpy as np

from cthree.Exceptions import IncompatibleQuantityException


class Quantity:
    """
    Represents any physical quantity used in the model or the pulse specification. For arithmetic operations just the
    numeric value is used. The value itself is stored in an optimizer friendly way as a float between -1 and 1. The
    conversion is given by
        scale * (value + 1) / 2 + offset

    Note on python's operators: equality checks `q == p` and `q != p` check for the values of the quantities q and p.
    For vector or matrix quantities, these check if all values are equal. If you want to be sure that two quantities
    are the same object (i.e. the same memory address), use `q is p`. Ordering operators like `q > p` will only work
    for scalar quantities and will raise an exception for vector or matrix quantities.

    Parameters
    ----------
    value: np.array(np.float64) or np.float64
        value of the quantity
    min_value: np.array(np.float64) or np.float64
        Minimum this quantity is allowed to take. If this is null, a default interval around the value will be chosen.
    max_value: np.array(np.float64) or np.float64
        Maximum this quantity is allowed to take. If this is null, a default interval around the value will be chosen.
    unit: str
        physical unit
    name: str
        symbol or description of this quantity
    twoPi: bool
        divide by two pi for representation
    """

    __unit: str
    __name: str
    __length: int
    __shape: Tuple
    # internal representation of the value
    __value: np.ndarray
    __offset: np.ndarray
    __scale: np.ndarray
    __twoPi: bool
    __dependent: bool
    __dependencies: List
    __relation: Callable
    __dependents: List

    def __init__(
            self,
            value: np.array,
            min_value: np.ndarray,
            max_value: np.ndarray,
            unit: str = "",
            name: str = "",
            twoPi: bool = False,
    ):
        if value is None or max_value is None or min_value is None:
            raise Exception("value, minimum, and maximum must be not null")

        self.__unit = unit
        self.__name = name
        self.__scale = np.array(0)
        self.__twoPi = twoPi

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

        self.setValue(value)

    @classmethod
    def relational(cls,
                   quantities: Quantity | List[Quantity],
                   relation: Callable,
                   unit: str | None = None,
                   name: str | None = None,
                   twoPi: bool = False) -> Self:
        """ Create a Quantity object that represents a Quantity that is calculated
        from other quantities using the relation function.

        Parameters
        ----------
        quantities: Quantity | List[Quantity]
            The quantities from which to calculate the value of self.
        relation: Callable
            Function describing how to calculate the value of self from other Quantities.
        unit: str | None
            The unit of the resulting Quantity. If 'None', then the units of all
            quantities are assumed the same.
        name: str | None
            A string identifier name of the resulting Quantity. If 'None', then
            a name is generated from the names of the related Quantities.
        twoPi: bool
            Divide by two pi for representation.

        Returns
        -------
        Quantity
            The Quantity with a relation set up, which recalculates the value of
            self from all dependencies.

        Raises
        ------
            ValueError:
                If any quantities do not have the same unit and no special unit is specified.
        """
        quantities = quantities if isinstance(quantities, List) else [quantities]
        min_val = np.min([qty.getMinValue() for qty in quantities])
        max_val = np.max([qty.getMaxValue() for qty in quantities])

        if name is None:
            name = 'relation_of'
            for qty in quantities:
                name += ('_' + qty.getName())

        if unit is None:
            if not all(qty.getUnit() == quantities[0].getUnit() for qty in quantities):
                raise ValueError(f"All quantities in creation on {name} "
                                 f"must have the same unit if no unit is specified.")
            unit = quantities[0].getUnit()

        qty = cls(value=min_val,
                  min_value=min_val,
                  max_value=max_val,
                  unit=unit,
                  name=name,
                  twoPi=twoPi)
        qty.addRelation(quantities, relation)
        return qty

    @classmethod
    def relationalCopy(cls, quantity: Quantity) -> Self:
        """ Create a Quantity object that is a one to one copy of a Quantity.
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

        qty = cls(value=quantity.getValue(),
                  min_value=quantity.getMinValue(),
                  max_value=quantity.getMaxValue(),
                  unit=quantity.getUnit(),
                  name=quantity.getName(),
                  twoPi=quantity.__twoPi)
        qty.addRelation(quantity, lambda x: x)
        return qty

    @property
    def dependent(self):
        """ The dependency status of the quantity
        if True:
            The value of this quantity is calculated from other quantities
        if False:
            The value of this quantity is independent of any other quantity
        """
        return self.__dependent

    def addRelation(self,
                    other: Quantity | List[Quantity],
                    relation: Callable,
                    checkUnits: bool = True) -> None:
        """ Adds a relation of self to one or more other quantities.

        Parameters
        ----------
        other: Quantity | List[Quantity]
            The quantities from which to calculate the value of self.
        relation: Callable
            Function describing how to calculate the value of self from other Quantities.
        checkUnits: bool
            If False, the check for equal units is not performed and unequal
            units are allowed.

        """
        other = other if isinstance(other, List) else [other]

        if not all(qty.getUnit() == self.getUnit() for qty in other) and checkUnits:
            raise ValueError(f"Not all Quantities in the relation have the same units. "
                             f"This may lead to unintentional physical errors. "
                             f"Set 'checkUnits=False' if this behavior is wanted.")

        self.__dependent = True
        self.__dependencies = other
        self.__relation = relation
        self.update()

        for qty in self.__dependencies:
            qty.__dependents.append(self)

    def update(self):
        """ Update function that is called if a value that this quantity is dependent on is changed."""
        self.__setValue(self.__relation(*[qty.getValue() for qty in self.__dependencies]))

    def getValue(self) -> np.array:
        return self.__scale * (self.__value + 1) / 2 + self.__offset

    def getReducedValue(self) -> np.ndarray:
        """
        Returns the value in the reduced representation as it is stored internally.
        """
        return np.reshape(self.__value, (-1, 1))

    def setValue(self, value) -> None:
        """
        Sets the value of this quantity. Value needs to be within the range of min_value and max_value.
        """
        if self.__dependent:
            raise ValueError("Cannot set value on dependent quantities, as it is calculated from other quantities.")

        self.__setValue(value)

    def __setValue(self, value) -> None:
        if isinstance(value, np.ndarray):
            val = value.astype(np.float64)
        else:
            val = np.array(value).astype(np.float64)
        tmp = 2 * (np.reshape(val, self.__shape) - self.__offset) / self.__scale - 1

        if np.any(np.abs(tmp) > 1.0):
            print("Error: ", val, self.getMinValue(), self.getMaxValue())
            raise ValueError(
                f"Value {self.__toString(val)}{self.__unit} out of bounds for quantity with "
                f"min_val: {self.__toString(self.getMinValue())}{self.__unit} and "
                f"max_val: {self.__toString(self.getMaxValue())}{self.__unit}",
            )
        self.__value = tmp

        # update all Quantities that depend on self
        for qty in self.__dependents:
            qty.update()

    def setReducedValue(self, value) -> None:
        if np.shape(value) == ():
            value = np.array([value])
        self.__value = value

    def getMinValue(self) -> np.ndarray:
        return self.__offset

    def getMaxValue(self) -> np.ndarray:
        return self.__scale + self.__offset

    def getScale(self) -> np.ndarray:
        return self.__scale

    def getLength(self) -> int:
        return self.__length

    def setLimits(self, min_value, max_value) -> None:
        """
        Sets the allowed minimum and maximum of this quantity.
        """
        oldValue = self.getValue()
        self.__offset = np.array(min_value)
        self.__scale = np.abs(np.array(max_value) - np.array(min_value))
        # the value is based on offset and scale and needs to be updated
        self.setValue(oldValue)

    def getName(self) -> str:
        """
        Returns the symbol or description or this quantity. Note that this does not have to be unique. For uniquely
        identifying a quantity, use getUUID.
        :return:
        """
        return self.__name

    def getUnit(self) -> str:
        return self.__unit

    def isScalar(self) -> bool:
        return self.__length == 1

    def isVector(self) -> bool:
        return self.__length > 1 and len(self.__shape) == 1

    # Python specific functions
    def __add__(self, other) -> Quantity:
        out_val = copy.deepcopy(self)
        out_val.setValue(self.getValue() + other)
        return out_val

    def __radd__(self, other) -> Quantity:
        out_val = copy.deepcopy(self)
        out_val.setValue(self.getValue() + other)
        return out_val

    def __sub__(self, other) -> Quantity:
        out_val = copy.deepcopy(self)
        out_val.setValue(self.getValue() - other)
        return out_val

    def __rsub__(self, other) -> Quantity:
        out_val = copy.deepcopy(self)
        out_val.setValue(other - self.getValue())
        return out_val

    def __mul__(self, other) -> Quantity:
        out_val = copy.deepcopy(self)
        out_val.setValue(self.getValue() * other)
        return out_val

    def __rmul__(self, other) -> Quantity:
        out_val = copy.deepcopy(self)
        out_val.setValue(self.getValue() * other)
        return out_val

    def __pow__(self, other) -> Quantity:
        out_val = copy.deepcopy(self)
        out_val.setValue(np.float_power(self.getValue(), other))
        return out_val

    def __rpow__(self, other) -> Quantity:
        out_val = copy.deepcopy(self)
        out_val.setValue(np.float_power(other, self.getValue()))
        return out_val

    def __truediv__(self, other) -> Quantity:
        out_val = copy.deepcopy(self)
        out_val.setValue(self.getValue() / other)
        return out_val

    def __rtruediv__(self, other) -> Quantity:
        out_val = copy.deepcopy(self)
        out_val.setValue(other / self.getValue())
        return out_val

    def __mod__(self, other) -> Quantity:
        out_val = copy.deepcopy(self)
        out_val.setValue(self.getValue() % other)
        return out_val

    def __lt__(self, other) -> bool:
        if not self.isScalar():
            raise IncompatibleQuantityException(
                "Ordering operators are only usable with scalar quantities"
            )
        return self.getValue() < other.getValue()

    def __le__(self, other) -> bool:
        if not self.isScalar():
            raise IncompatibleQuantityException(
                "Ordering operators are only usable with scalar quantities"
            )
        return self.getValue() <= other

    def __eq__(self, other) -> bool:
        if self.__shape != other.__shape:
            return False
        return all(self.getValue() == other)

    def __ne__(self, other) -> bool:
        if self.__shape != other.__shape:
            return True
        return any(self.getValue() != other)

    def __ge__(self, other) -> bool:
        if not self.isScalar():
            raise IncompatibleQuantityException(
                "Ordering operators are only usable with scalar quantities"
            )
        return self.getValue() >= other

    def __gt__(self, other) -> bool:
        if not self.isScalar():
            raise IncompatibleQuantityException(
                "Ordering operators are only usable with scalar quantities"
            )
        return self.getValue() > other

    def __array__(self):
        return np.array(self.getValue())

    def __len__(self):
        return self.__length

    def __getitem__(self, key):
        if self.__length == 1 and key == 0:
            return self.getValue()
        return self.getValue().__getitem__(key)

    def __abs__(self):
        return abs(self.getValue())

    def __float__(self):
        if self.__length > 1:
            raise NotImplementedError
        return float(np.squeeze(self.getValue()))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.__toString(self.getValue())

    def __toString(self, val):
        ret = ""
        for entry in np.nditer(val):
            if self.__unit != "":
                if self.__twoPi:
                    ret += (
                            self.__makeHumanReadable(entry / np.pi / 2)
                            + self.__unit
                            + " x 2pi "
                    )
                else:
                    ret += self.__makeHumanReadable(entry) + self.__unit + " "
            else:
                if self.__twoPi:
                    ret += (
                            self.__makeHumanReadable(entry / np.pi / 2, use_prefix=False)
                            + " x 2pi "
                    )
                else:
                    ret += self.__makeHumanReadable(entry, use_prefix=False) + " "
        if self.__name:
            ret = self.__name + ": " + ret
        return ret

    @staticmethod
    def __makeHumanReadable(val, use_prefix: bool = True) -> str:
        """
        Converts a number to a human readable string in engineering notation.
        """
        if use_prefix:
            num, prefix = Quantity.__engineeringNumber(val)
            formatted_string = f"{num:.3g} " + prefix
        else:
            formatted_string = f"{val:.3g} "
        return formatted_string

    # Internal utility functions
    @staticmethod
    def __engineeringNumber(val: float) -> Tuple[float, str]:
        """
        Converts a number to engineering notation by returning number and prefix.
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
