import pytest
from typing import Tuple

from cthree.Quantity import Quantity
import numpy as np
import numpy.testing as testing


@pytest.fixture
def five():
    return Quantity(5, 0, 15)


@pytest.fixture
def three():
    return Quantity(3, 0, 15)


# getter and setter


def testGet(randomQuantityForValues) -> None:
    """
    Tests get_value, get_min_value, and get_max_value
    """
    for N in range(1, 100):
        # create a quantity with random values and check if the get functions return the same values
        values = (2 * np.random.random(N) - 1) * np.power(
            10.0, np.random.randint(-10, 10)
        )
        q = randomQuantityForValues(values)
        testing.assert_allclose(q.getValue(), values)
        testing.assert_array_less(q.getMinValue(), q.getValue())
        testing.assert_array_less(q.getValue(), q.getMaxValue())


def testSet(randomQuantityForValues, randomLimitsForQuantity) -> None:
    """
    Tests setValue, setMinValue, and setMaxValue
    """
    for N in range(1, 100):
        # create a random quantity with values that will be overwritten
        values = (2 * np.random.random(N) - 1) * np.power(
            10.0, np.random.randint(-10, 10)
        )
        q = randomQuantityForValues(values)
        oldMin = q.getMinValue()
        oldMax = q.getMaxValue()

        # set new values within the limits and check that the limits stay unchanged
        newValues = np.random.random(N) * (oldMax - oldMin) + oldMin
        q.setValue(newValues)
        testing.assert_allclose(q.getValue(), newValues)
        testing.assert_allclose(q.getMinValue(), oldMin)
        testing.assert_allclose(q.getMaxValue(), oldMax)

        # set new limits and check that the values stay unchanged
        newLimits = randomLimitsForQuantity(newValues)
        q.setLimits(*newLimits)
        testing.assert_allclose(q.getMinValue(), newLimits[0])
        testing.assert_allclose(q.getMaxValue(), newLimits[1])
        testing.assert_array_less(q.getMinValue(), q.getValue())
        testing.assert_array_less(q.getValue(), q.getMaxValue())


def testGetItem(randomQuantityForValues) -> None:
    for N in range(1, 100):
        values = (2 * np.random.random(N) - 1) * np.power(
            10.0, np.random.randint(-10, 10)
        )
        q = randomQuantityForValues(values)
        for i in range(len(q)):
            testing.assert_almost_equal(q[i], values[i])


def testLen(randomQuantityForValues) -> None:
    for N in range(1, 100):
        values = (2 * np.random.random(N) - 1) * np.power(
            10.0, np.random.randint(-10, 10)
        )
        q = randomQuantityForValues(values)
        testing.assert_almost_equal(len(q), N)


# conversion
def testFloat(randomQuantityForValues) -> None:
    for i in range(100):
        value = (2 * np.random.random(1) - 1) * np.power(
            10.0, np.random.randint(-10, 10)
        )
        q = randomQuantityForValues(value)
        testing.assert_almost_equal(float(q), value)


def testToArray(randomQuantityForValues) -> None:
    for N in range(1, 100):
        values = (2 * np.random.random(N) - 1) * np.power(
            10.0, np.random.randint(-10, 10)
        )
        q = randomQuantityForValues(values)
        testing.assert_array_almost_equal(np.array(q), values)


# comparison
def testComparisons(randomQuantityForValues) -> None:
    """
    Tests __lt__, __le__, __gt__, and __ge__ for scalar quantities.
    """
    for i in range(1, 100):
        # create a quantity with random values and check if the get functions return the same values
        value = (2 * np.random.random() - 1) * np.power(
            10.0, np.random.randint(-10, 10)
        )
        q1 = randomQuantityForValues(value)
        q2 = randomQuantityForValues(2 * np.abs(value))

        assert q2 > q1
        assert q2 >= q1
        assert q1 < q2
        assert q1 <= q2
        assert q1 <= q1
        assert q1 >= q1


def testEquality(randomQuantityForValues) -> None:
    """
    Tests __eq__ and __ne__ for scalar quantities.
    """
    for i in range(1, 100):
        # create a quantity with random values and check if the get functions return the same values
        value = (2 * np.random.random() - 1) * np.power(
            10.0, np.random.randint(-10, 10)
        )
        q1 = randomQuantityForValues(value)
        q2 = randomQuantityForValues(1.2 * np.abs(value))

        assert q1 != q2
        assert q2 != q1
        assert q1 == q1
        assert q2 == q2
        testing.assert_almost_equal(q1, value)
        testing.assert_almost_equal(q1, q1.getValue())


def testEqualityById(randomQuantity):
    q1 = randomQuantity(1)
    q2 = Quantity(q1.getValue(), q1.getMinValue(), q1.getMaxValue(), q1.getUnit())
    assert q1 == q2
    assert q1 is not q2
    assert q2 is not q1
    assert q1 is q1
    assert q2 is q2


def testNoInput():
    """
    Trying to instantiate without any parameters.
    """
    with pytest.raises(Exception):
        Quantity(np.random.random())
    with pytest.raises(Exception):
        Quantity()
    with pytest.raises(Exception):
        Quantity(np.random.random(), min_value=np.random.random(), max_value=None)
    with pytest.raises(Exception):
        Quantity(np.random.random(), min_value=None, max_value=np.random.random())


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
    volts = Quantity(0.005, 0, 1, unit="V")
    assert str(volts) == "5.0 mV "
    resist = Quantity(2100, 0, 2500, unit="Ohm")
    assert str(resist) == "2.1 KOhm "


def testIsScalarOrVector(randomQuantity):
    for i in range(20):
        q = randomQuantity(1)
        assert q.isScalar()
        assert not q.isVector()

    for dim in range(2, 100):
        for i in range(20):
            q = randomQuantity(dim)
            assert q.isVector()
            assert not q.isScalar()

            v = __generateRandomMatrix(dim)
            q2 = Quantity(v, v - 1, v + 1)
            assert not q2.isVector()
            assert not q2.isScalar()


def __generateRandomMatrix(N: int) -> np.ndarray:
    magnitude = np.power(10.0, np.random.randint(-10, 10))
    return (2 * np.random.random((N, N)) - 1) * magnitude
