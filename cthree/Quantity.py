from __future__ import annotations  # necessary for type hints

import copy
from typing import Tuple, List

import numpy as np


# TODO: change from numpy to tensorflow / jax
# TODO: change from direct storage to reduced units using scale and offset
# TODO: docstrings

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
        minimum this quantity is allowed to take
    max_value: np.array(np.float64) or np.float64
        maximum this quantity is allowed to take
    unit: str
        physical unit
    """
    __value: np.array
    __min_value: np.array
    __max_value: np.array
    __unit: str
    __length: int
    __shape: Tuple

    def __init__(self, value: np.array, min_value: np.array, max_value: np.array, unit: str = None):
        self.__value = value
        self.__min_value = min_value
        self.__max_value = max_value
        self.__unit = unit

        value = np.array(value)
        if hasattr(value, "shape"):
            self.__shape = value.shape
            self.__length = int(np.prod(value.shape))
        else:
            self.__shape = (1,)
            self.__length = 1

    # Getter and setter functions
    def get_value(self) -> np.array:
        return self.__value

    def set_value(self, value) -> None:
        self.__value = value

    def get_min_value(self) -> np.array:
        return self.__min_value

    def get_max_value(self) -> np.array:
        return self.__max_value

    def set_limits(self, min_value, max_value) -> None:
        self.__min_value = min_value
        self.__max_value = max_value

    # Python specific functions
    def __add__(self, other) -> Quantity:
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() + other)
        return out_val

    def __radd__(self, other) -> Quantity:
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() + other)
        return out_val

    def __sub__(self, other) -> Quantity:
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() - other)
        return out_val

    def __rsub__(self, other) -> Quantity:
        out_val = copy.deepcopy(self)
        out_val.set_value(other - self.get_value())
        return out_val

    def __mul__(self, other) -> Quantity:
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() * other)
        return out_val

    def __rmul__(self, other) -> Quantity:
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() * other)
        return out_val

    def __pow__(self, other) -> Quantity:
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() ** other)
        return out_val

    def __rpow__(self, other) -> Quantity:
        out_val = copy.deepcopy(self)
        out_val.set_value(other ** self.get_value())
        return out_val

    def __truediv__(self, other) -> Quantity:
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() / other)
        return out_val

    def __rtruediv__(self, other) -> Quantity:
        out_val = copy.deepcopy(self)
        out_val.set_value(other / self.get_value())
        return out_val

    def __mod__(self, other) -> Quantity:
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() % other)
        return out_val

    def __lt__(self, other) -> bool:
        return self.get_value() < other

    def __le__(self, other) -> bool:
        return self.get_value() <= other

    def __eq__(self, other) -> bool:
        return self.get_value() == other

    def __ne__(self, other) -> bool:
        return self.get_value() != other

    def __ge__(self, other) -> bool:
        return self.get_value() >= other

    def __gt__(self, other) -> bool:
        return self.get_value() > other

    def __array__(self):
        return np.array(self.get_value())

    def __len__(self):
        return self.__length

    def __getitem__(self, key):
        if self.__length == 1 and key == 0:
            return self.get_value()
        return self.get_value().__getitem__(key)

    def __float__(self):
        if self.__length > 1:
            raise NotImplementedError
        return float(self.get_value())

    def __repr__(self):
        return self.__str__()[:-1]

    def __str__(self):
        val = self.get_value()
        ret = ""
        for entry in np.nditer(val):
            if self.__unit is None:
                ret += f"{entry:.3} "
            else:
                num, prefix = self.__engineeringNumber(entry)
                ret += f"{entry:.3f} " + prefix + self.__unit + " "
        return ret

    # Internal utility functions
    @staticmethod
    def __engineeringNumber(val: float) -> Tuple[float, str]:
        """
        Converts a number to engineering notation by returning number and prefix.
        """
        if np.isnan(val):
            return np.nan, "NaN"

        sign = 1
        if val == 0:
            return 0, ""
        if val < 0:
            val = -val
            sign = -1
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
