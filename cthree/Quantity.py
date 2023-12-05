from __future__ import annotations  # necessary for type hints

import copy
from typing import Tuple

import numpy as np


class Quantity:
    """
    Represents any physical quantity used in the model or the pulse specification. For arithmetic operations just the
    numeric value is used. The value itself is stored in an optimizer friendly way as a float between -1 and 1. The
    conversion is given by
        scale * (value + 1) / 2 + offset

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
    """

    __unit: str
    __name: str
    __length: int
    __shape: Tuple
    # internal representation of the value
    __value: np.ndarray
    __offset: np.ndarray
    __scale: np.ndarray

    def __init__(
        self,
        value: np.array,
        min_value: np.ndarray,
        max_value: np.ndarray,
        unit: str = "",
        name: str = "",
    ):
        if value is None or max_value is None or min_value is None:
            raise Exception("value, minimum, and maximum must be not null")

        self.__unit = unit
        self.__name = name
        self.__scale = 0

        value = np.array(value)
        if hasattr(value, "shape"):
            self.__shape = value.shape
            self.__length = int(np.prod(value.shape))
        else:
            self.__shape = (1,)
            self.__length = 1

        self.__offset = np.array(min_value)
        self.__scale = np.abs(np.array(max_value) - np.array(min_value))
        self.setValue(value)

    # Getter and setter functions
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
        if isinstance(value, np.ndarray):
            val = value.astype(np.float64)
        else:
            val = np.array(value, np.float64)
        tmp = 2 * (np.reshape(val, self.__shape) - self.__offset) / self.__scale - 1

        if np.any(np.abs(tmp) > 1.0):
            print("Error: ", val, self.getMinValue(), self.getMaxValue())
            raise ValueError(
                f"Value {self.__toString(val)}{self.__unit} out of bounds for quantity with "
                f"min_val: {self.__toString(self.getMinValue())}{self.__unit} and "
                f"max_val: {self.__toString(self.getMaxValue())}{self.__unit}",
            )
        self.__value = tmp

    def setReducedValue(self, value) -> None:
        self.__value = value

    def getMinValue(self) -> np.ndarray:
        return self.__offset

    def getMaxValue(self) -> np.ndarray:
        return self.__scale + self.__offset

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

    def getUnit(self) -> str:
        return self.__unit

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
        return self.getValue() < other

    def __le__(self, other) -> bool:
        return self.getValue() <= other

    def __eq__(self, other) -> bool:
        return self.getValue() == other

    def __ne__(self, other) -> bool:
        return self.getValue() != other

    def __ge__(self, other) -> bool:
        return self.getValue() >= other

    def __gt__(self, other) -> bool:
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
        return float(self.getValue())

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.__toString(self.getValue())

    def __toString(self, val):
        ret = ""
        for entry in np.nditer(val):
            if self.__unit != "":
                ret += self.__makeHumanReadable(entry) + self.__unit + " "
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
            formatted_string = f"{num:.3} " + prefix
        else:
            formatted_string = f"{val:.3} "
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
