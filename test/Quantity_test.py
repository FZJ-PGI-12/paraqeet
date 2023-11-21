import pytest
from typing import Tuple

from cthree.Quantity import Quantity
import numpy as np
import numpy.testing as testing

"""
properties: __repr__, __str__
arithmetic: add, subtract
"""


@pytest.fixture
def five():
    return Quantity(5, 0, 15)


@pytest.fixture
def three():
    return Quantity(3, 0, 15)


# getter and setter


def testGet() -> None:
    """
    Tests get_value, get_min_value, and get_max_value
    """
    for N in range(1, 100):
        # create a quantity with random values and check if the get functions return the same values
        values = (2 * np.random.random(N) - 1) * np.power(
            10.0, np.random.randint(-10, 10)
        )
        q = __generateQuantity(values)
        testing.assert_allclose(q.getValue(), values)
        testing.assert_array_less(q.getMinValue(), q.getValue())
        testing.assert_array_less(q.getValue(), q.getMaxValue())


def testSet() -> None:
    """
    Tests setValue, setMinValue, and setMaxValue
    """
    for N in range(1, 100):
        # create a random quantity with values that will be overwritten
        values = (2 * np.random.random(N) - 1) * np.power(
            10.0, np.random.randint(-10, 10)
        )
        q = __generateQuantity(values)
        oldMin = q.getMinValue()
        oldMax = q.getMaxValue()

        # set new values within the limits and check that the limits stay unchanged
        newValues = np.random.random(N) * (oldMax - oldMin) + oldMin
        q.setValue(newValues)
        testing.assert_allclose(q.getValue(), newValues)
        testing.assert_allclose(q.getMinValue(), oldMin)
        testing.assert_allclose(q.getMaxValue(), oldMax)

        # set new limits and check that the values stay unchanged
        newLimits = __generateRandomLimits(newValues)
        q.setLimits(*newLimits)
        testing.assert_allclose(q.getMinValue(), newLimits[0])
        testing.assert_allclose(q.getMaxValue(), newLimits[1])
        testing.assert_array_less(q.getMinValue(), q.getValue())
        testing.assert_array_less(q.getValue(), q.getMaxValue())


def testGetItem() -> None:
    for N in range(1, 100):
        values = (2 * np.random.random(N) - 1) * np.power(
            10.0, np.random.randint(-10, 10)
        )
        q = __generateQuantity(values)
        for i in range(len(q)):
            testing.assert_almost_equal(q[i], values[i])


def testLen() -> None:
    for N in range(1, 100):
        values = (2 * np.random.random(N) - 1) * np.power(
            10.0, np.random.randint(-10, 10)
        )
        q = __generateQuantity(values)
        testing.assert_almost_equal(len(q), N)


# conversion
def testFloat() -> None:
    for i in range(100):
        value = (2 * np.random.random(1) - 1) * np.power(
            10.0, np.random.randint(-10, 10)
        )
        q = __generateQuantity(value)
        testing.assert_almost_equal(float(q), value)


def testToArray() -> None:
    for N in range(1, 100):
        values = (2 * np.random.random(N) - 1) * np.power(
            10.0, np.random.randint(-10, 10)
        )
        q = __generateQuantity(values)
        testing.assert_array_almost_equal(np.array(q), values)


# comparison
def testComparisons() -> None:
    """
    Tests __lt__, __le__, __gt__, and __ge__ for scalar quantities.
    """
    for i in range(1, 100):
        # create a quantity with random values and check if the get functions return the same values
        value = (2 * np.random.random() - 1) * np.power(
            10.0, np.random.randint(-10, 10)
        )
        q1 = __generateQuantity(value)
        q2 = __generateQuantity(2 * np.abs(value))

        assert q2 > q1
        assert q2 >= q1
        assert q1 < q2
        assert q1 <= q2
        assert q1 <= q1
        assert q1 >= q1


def testEquality() -> None:
    """
    Tests __eq__ and __ne__ for scalar quantities.
    """
    for i in range(1, 100):
        # create a quantity with random values and check if the get functions return the same values
        value = (2 * np.random.random() - 1) * np.power(
            10.0, np.random.randint(-10, 10)
        )
        q1 = __generateQuantity(value)
        q2 = __generateQuantity(1.2 * np.abs(value))

        assert q1 != q2
        assert q2 != q1
        assert q1 == q1
        assert q2 == q2
        testing.assert_almost_equal(q1, value)
        testing.assert_almost_equal(q1, q1.getValue())


def testNoInput():
    """
    Trying to instantiate without any parameters.
    """
    with pytest.raises(Exception):
        Quantity(5)


def testOutOfBounds():
    num = Quantity(5, 3, 6)
    with pytest.raises(Exception):
        num.setValue(7)


def testArithmetic(five, three):
    testing.assert_almost_equal(five + three, 8)
    testing.assert_almost_equal(5 + three, 8)
    testing.assert_almost_equal(five - three, 2)
    testing.assert_almost_equal(five * three, 15)
    testing.assert_almost_equal(five / three, 5.0 / 3)
    testing.assert_almost_equal(5 / three, 5.0 / 3)
    testing.assert_almost_equal(five % 3, 5.0 % 3)
    testing.assert_almost_equal(five * 3, 15)
    testing.assert_almost_equal(5 * three, 15)
    testing.assert_almost_equal(five**1, 5)
    testing.assert_almost_equal(1**three, 1)


def testStr(five):
    str(five) == "5"
    volts = Quantity(0.005, 0, 1, unit="V")
    assert str(volts) == "5.0 mV "
    resist = Quantity(2100, 0, 2500, unit="Ohm")
    assert str(resist) == "2.1 KOhm "


# helper functions
def __generateRandomQuantity(N: int) -> Quantity:
    """
    Generates a quantity with N positive and negative numbers, each with the same order of magnitude which is chosen
    randomly between 1e-10 and 1e10.
    """
    magnitude = np.power(10.0, np.random.randint(-10, 10))
    values = (2 * np.random.random(N) - 1) * magnitude
    return __generateQuantity(values)


def __generateQuantity(values: np.array) -> Quantity:
    """
    Generates a quantity from the given array of values, making sure that the limits are set correctly.
    """
    limits = __generateRandomLimits(values)
    return Quantity(values, min_value=limits[0], max_value=limits[1], unit="")


def __generateRandomLimits(values: np.array) -> Tuple:
    """
    Returns random but valid minimum and maximum values for the given value array while taking acount for negative
    values.
    """
    if len(values.shape) == 0:
        # scalar quantity
        if values == 0.0:
            min_value = -1
            max_value = +1
        elif values < 0:
            min_value = (np.random.random() + 1) * values
            max_value = np.random.random() * values
        else:
            min_value = np.random.random() * values
            max_value = (np.random.random() + 1) * values
        return min_value, max_value
    else:
        # list quantity
        min_values, max_values = np.zeros_like(values), np.zeros_like(values)
        for i, v in enumerate(values):
            if v == 0.0:
                min_values[i] = -1
                max_values[i] = +1
            elif v < 0:
                min_values[i] = (np.random.random() + 1) * v
                max_values[i] = np.random.random() * v
            else:
                min_values[i] = np.random.random() * v
                max_values[i] = (np.random.random() + 1) * v
        return min_values, max_values
