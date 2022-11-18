import copy

from cthree.Quantity import Quantity
import numpy as np
import numpy.testing as testing

'''
properties: __repr__, __str__
arithmetic: add, subtract
'''


# getter and setter

def testGet() -> None:
    """
    Tests get_value, get_min_value, and get_max_value
    """
    for N in range(1, 100):
        # create a quantity with random values and check if the get functions return the same values
        values = (2 * np.random.random(N) - 1) * np.power(10.0, np.random.randint(-10, 10))
        q = __generateQuantity(values)
        testing.assert_allclose(q.get_value(), values)
        testing.assert_array_less(q.get_min_value(), q.get_value())
        testing.assert_array_less(q.get_value(), q.get_max_value())


def testSet() -> None:
    """
    Tests set_value, set_min_value, and set_max_value
    """
    for N in range(1, 100):
        # create a random quantity with values that will be overwritten
        values = (2 * np.random.random(N) - 1) * np.power(10.0, np.random.randint(-10, 10))
        q = __generateQuantity(values)
        oldMin = q.get_min_value()
        oldMax = q.get_max_value()

        # set new values and check that the limits stay unchanged
        newValues = np.random.random(N) * np.power(10.0, np.random.randint(-10, 10))
        q.set_value(newValues)
        testing.assert_allclose(q.get_value(), newValues)
        testing.assert_allclose(q.get_min_value(), oldMin)
        testing.assert_allclose(q.get_max_value(), oldMax)

        # set new limits and check that the values stay unchanged
        q.set_limits(0.5 * newValues, 1.5 * newValues)
        testing.assert_allclose(q.get_min_value(), 0.5 * newValues)
        testing.assert_allclose(q.get_max_value(), 1.5 * newValues)
        testing.assert_array_less(q.get_min_value(), q.get_value())
        testing.assert_array_less(q.get_value(), q.get_max_value())


def testGetItem() -> None:
    for N in range(1, 100):
        values = (2 * np.random.random(N) - 1) * np.power(10.0, np.random.randint(-10, 10))
        q = __generateQuantity(values)
        for i, v in enumerate(q):
            assert (v == values[i])


def testLen() -> None:
    for N in range(1, 100):
        values = (2 * np.random.random(N) - 1) * np.power(10.0, np.random.randint(-10, 10))
        q = __generateQuantity(values)
        assert (len(q) == N)


# conversion
def testFloat() -> None:
    for i in range(100):
        value = (2 * np.random.random(1) - 1) * np.power(10.0, np.random.randint(-10, 10))
        q = __generateQuantity(value)
        assert (float(q) == value)


def testToArray() -> None:
    for N in range(1, 100):
        values = (2 * np.random.random(N) - 1) * np.power(10.0, np.random.randint(-10, 10))
        q = __generateQuantity(values)
        testing.assert_array_almost_equal(np.array(q), values)


# comparison
def testComparisons() -> None:
    """
    Tests __lt__, __le__, __gt__, and __ge__ for scalar quantities.
    """
    for i in range(1, 100):
        # create a quantity with random values and check if the get functions return the same values
        value = (2 * np.random.random() - 1) * np.power(10.0, np.random.randint(-10, 10))
        q1 = __generateQuantity(value)
        q2 = __generateQuantity(2 * np.abs(value))

        assert (q2 > q1)
        assert (q2 >= q1)
        assert (q1 < q2)
        assert (q1 <= q2)
        assert (q1 <= q1)
        assert (q1 >= q1)


def testEquality() -> None:
    """
    Tests __eq__ and __ne__ for scalar quantities.
    """
    for i in range(1, 100):
        # create a quantity with random values and check if the get functions return the same values
        value = (2 * np.random.random() - 1) * np.power(10.0, np.random.randint(-10, 10))
        q1 = __generateQuantity(value)
        q2 = __generateQuantity(2 * np.abs(value))

        assert (q1 != q2)
        assert (q2 != q1)
        assert (q1 == value)
        assert (q1 == q1.get_value())
        assert (q1 == copy.deepcopy(q1))


# arithmetic
def testArithmetic() -> None:
    """
    Tests __add__, __radd__, __sub__, __rsub__, __mul__, __rmul__, __truediv__, and __rtruediv__.
    """
    for N in range(1, 100):
        values = (2 * np.random.random(N) - 1) * np.power(10.0, np.random.randint(-10, 10))
        q = __generateQuantity(values)

        # arithmetic with a Quantity and a scalar
        scalar = np.random.random()
        testing.assert_array_almost_equal(q + scalar, values + scalar)
        testing.assert_array_almost_equal(q - scalar, values - scalar)
        testing.assert_array_almost_equal(scalar + q, scalar + values)
        testing.assert_array_almost_equal(scalar - q, scalar - values)

        testing.assert_array_almost_equal(q * scalar, values * scalar)
        testing.assert_array_almost_equal(q / scalar, values / scalar)
        testing.assert_array_almost_equal(scalar * q, scalar * values)
        testing.assert_array_almost_equal(scalar / q, scalar / values)

        # arithmetic with a Quantity and an array
        array = (2 * np.random.random(N) - 1) * np.power(10.0, np.random.randint(-10, 10))
        testing.assert_array_almost_equal(q + array, values + array)
        testing.assert_array_almost_equal(q - array, values - array)
        testing.assert_array_almost_equal(array + q, array + values)
        testing.assert_array_almost_equal(array - q, array - values)

        testing.assert_array_almost_equal(q * array, values * array)
        testing.assert_array_almost_equal(q / array, values / array)
        testing.assert_array_almost_equal(array * q, array * values)
        testing.assert_array_almost_equal(array / q, array / values)


def testPow() -> None:
    """
    Tests __pow__ and __rpow__.
    """
    for N in range(1, 100):
        values = (2 * np.random.random(N) - 1) * np.power(10.0, np.random.randint(-3, 3))
        q = __generateQuantity(values)

        # Quantity and scalar
        scalar = np.random.randint(1, 20, size=1)
        testing.assert_array_almost_equal(q ** scalar, values ** scalar)
        testing.assert_array_almost_equal(scalar ** q, scalar ** values)

        # Quantity and array
        array = np.random.randint(1, 20, size=N)
        testing.assert_array_almost_equal(q ** array, values ** array)
        testing.assert_array_almost_equal(array ** q, array ** values)


def testMod() -> None:
    """
    Tests __mod__.
    """
    for N in range(1, 100):
        values = (2 * np.random.random(N) - 1) * np.power(10.0, np.random.randint(-3, 3))
        q = __generateQuantity(values)

        scalar = np.random.randint(1, 20, size=1)
        testing.assert_array_almost_equal(q % scalar, values % scalar)
        array = np.random.randint(1, 20, size=N)
        testing.assert_array_almost_equal(q % array, values % array)


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
    if len(values.shape) == 0:
        # scalar quantity
        if values == 0.0:
            min_value = -1
            max_value = +1
        elif values < 0:
            min_value = 1.5 * values
            max_value = 0.5 * values
        else:
            min_value = 0.5 * values
            max_value = 1.5 * values
        return Quantity(values, min_value=min_value, max_value=max_value, unit="")
    else:
        # list quantity
        min_values, max_values = np.zeros_like(values), np.zeros_like(values)
        for i, v in enumerate(values):
            if v == 0.0:
                min_values[i] = -1
                max_values[i] = +1
            elif v < 0:
                min_values[i] = 1.5 * v
                max_values[i] = 0.5 * v
            else:
                min_values[i] = 0.5 * v
                max_values[i] = 1.5 * v

        return Quantity(values, min_value=min_values, max_value=max_values, unit="")


testGet()
testSet()
testGetItem()
testLen()
testFloat()
testToArray()
testComparisons()
testEquality()
testArithmetic()
testPow()
testMod()
