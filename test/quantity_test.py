"""Test the quantity object."""

import numpy as np
import numpy.testing as testing
import pytest

from cthree.exceptions import IncompatibleQuantityException
from cthree.quantity import Quantity


@pytest.fixture
def five():
    """Create a quantity with value '5' and a max value of '15'."""
    return Quantity(5, 0, 15)


@pytest.fixture
def three():
    """Create a quantity with value '3' and a max value of '15'."""
    return Quantity(3, 0, 15)


# constructor
def test_constructor_keeps_value_length():
    """If value, min, and max are arrays of length N, they should not be altered in the constructor."""
    for N in range(1, 100):
        values = np.random.rand(N)
        q = Quantity(values, min_value=0.9 * values, max_value=1.1 * values, unit="")
        assert len(q.get_value()) == N
        assert len(q.get_min_value()) == N
        assert len(q.get_max_value()) == N


def test_updates_bounds_length():
    """If value is of length N while min, and max are scalar, the bounds should be altered into arrays."""
    for N in range(1, 100):
        # In the constructor
        values = np.random.rand(N)
        q = Quantity(values, min_value=0.9 * np.min(values), max_value=1.1 * np.max(values), unit="")
        assert len(q.get_value()) == N

        minValue = q.get_min_value()
        assert len(minValue) == N
        testing.assert_allclose(minValue, minValue[0])

        maxValue = q.get_max_value()
        assert len(maxValue) == N
        testing.assert_allclose(maxValue, maxValue[0])

        # In set_limits
        q.set_limits(0.9 * np.min(values), 1.1 * np.max(values))
        minValue = q.get_min_value()
        assert len(minValue) == N
        testing.assert_allclose(minValue, minValue[0])

        maxValue = q.get_max_value()
        assert len(maxValue) == N
        testing.assert_allclose(maxValue, maxValue[0])


def test_fails_on_different_lengths(random_from_list):
    """If value, min, and max are arrays of different length, the quantity should raise an exception."""
    for N in range(1, 100):
        values = np.random.rand(N)
        # Test for 20 random shapes other than N if set_value, set_reduced_value, and set_value_and_limits fail
        for _ in range(20):
            minShape = random_from_list(np.arange(100), N)
            minValues = 0.9 * np.random.rand(minShape) * np.min(values)
            maxShape = random_from_list(np.arange(100), N)
            maxValues = 1.1 * np.random.rand(maxShape) * np.max(values)

            # In the constructor
            with pytest.raises(IncompatibleQuantityException):
                Quantity(values, min_value=minValues, max_value=maxValues, unit="")

            # In set_limits
            q = Quantity(values, min_value=0.9 * values, max_value=1.1 * values, unit="")
            with pytest.raises(IncompatibleQuantityException):
                q.set_limits(minValues, maxValues)
            testing.assert_array_equal(q.get_value(), values)
            testing.assert_almost_equal(q.get_min_value(), 0.9 * values)
            testing.assert_almost_equal(q.get_max_value(), 1.1 * values)

            # In set_value_and_limits
            with pytest.raises(IncompatibleQuantityException):
                q.set_value_and_limits(values, minValues, maxValues)
            testing.assert_array_equal(q.get_value(), values)
            testing.assert_almost_equal(q.get_min_value(), 0.9 * values)
            testing.assert_almost_equal(q.get_max_value(), 1.1 * values)


# getter and setter
def testGet(random_quantity_for_values) -> None:
    """Test get_value, get_min_value, and get_max_value.

    For scalar quantities.

    """
    for N in range(1, 100):
        # create a quantity with random values and check if the
        # get functions return the same values
        values = (2 * np.random.random(N) - 1) * np.power(10.0, np.random.randint(-10, 10))
        q = random_quantity_for_values(values)
        testing.assert_allclose(q.get_value(), values)
        testing.assert_array_less(q.get_min_value(), q.get_value())
        testing.assert_array_less(q.get_value(), q.get_max_value())


def testSet(random_quantity_for_values, random_limits_for_quantity) -> None:
    """Test setValue, setMinValue, and setMaxValue.

    For scalar quantities.

    """
    for N in range(1, 100):
        # create a random quantity with values that will be overwritten
        values = (2 * np.random.random(N) - 1) * np.power(10.0, np.random.randint(-10, 10))
        q = random_quantity_for_values(values)
        oldMin = q.get_min_value()
        oldMax = q.get_max_value()

        # set new values within the limits and check that the
        # limits stay unchanged
        newValues = np.random.random(N) * (oldMax - oldMin) + oldMin
        q.set_value(newValues)
        testing.assert_allclose(q.get_value(), newValues)
        testing.assert_allclose(q.get_min_value(), oldMin)
        testing.assert_allclose(q.get_max_value(), oldMax)

        # set new limits and check that the values stay unchanged
        newLimits = random_limits_for_quantity(newValues)
        q.set_limits(*newLimits)
        testing.assert_allclose(q.get_min_value(), newLimits[0])
        testing.assert_allclose(q.get_max_value(), newLimits[1])
        testing.assert_array_less(q.get_min_value(), q.get_value())
        testing.assert_array_less(q.get_value(), q.get_max_value())


def test_set_fails_on_different_lengths(random_quantity_for_values, random_from_list):
    """Setting a new value should fail if the length of the new value is different than the old one."""
    for N in range(1, 100):
        # create a random quantity with values that shall be overwritten
        values = (2 * np.random.random(N) - 1) * np.power(10.0, np.random.randint(-10, 10))
        q = random_quantity_for_values(values)
        oldValue = q.get_value()
        oldMin = q.get_min_value()
        oldMax = q.get_max_value()

        # Test for 20 random shapes other than N if set_value, set_reduced_value, and set_value_and_limits fail
        for _ in range(20):
            newShape = random_from_list(np.arange(100), N)
            newValues = np.random.rand(newShape)
            with pytest.raises(IncompatibleQuantityException):
                q.set_value(newValues)
            with pytest.raises(IncompatibleQuantityException):
                q.set_reduced_value(newValues)
            if N > 1:
                # If the bounds have a shape of N=1, set_value_and_limits would correctly assume that the bounds should
                # be transformed from scalar into array
                with pytest.raises(IncompatibleQuantityException):
                    q.set_value_and_limits(newValues, oldMin, oldMax)

        # Make sure that the quantity did not change
        testing.assert_array_equal(q.get_value(), oldValue)
        testing.assert_array_equal(q.get_min_value(), oldMin)
        testing.assert_array_equal(q.get_max_value(), oldMax)


def testGetItem(random_quantity_for_values) -> None:
    """Test get item for scalar quantities."""
    for N in range(1, 100):
        values = (2 * np.random.random(N) - 1) * np.power(10.0, np.random.randint(-10, 10))
        q = random_quantity_for_values(values)
        for i in range(len(q)):
            testing.assert_almost_equal(q[i], values[i])


def testLen(random_quantity_for_values) -> None:
    """Test the length value for scalar quantities."""
    for N in range(1, 100):
        values = (2 * np.random.random(N) - 1) * np.power(10.0, np.random.randint(-10, 10))
        q = random_quantity_for_values(values)
        testing.assert_almost_equal(len(q), N)


# conversion
def testFloat(random_quantity_for_values) -> None:
    """Test float conversion for scalar quantities."""
    for i in range(100):
        value = (2 * np.random.random(1) - 1) * np.power(10.0, np.random.randint(-10, 10))
        q = random_quantity_for_values(value)
        testing.assert_almost_equal(float(q), value)


def testToArray(random_quantity_for_values) -> None:
    """Test array conversion for scalar quantities."""
    for N in range(1, 100):
        values = (2 * np.random.random(N) - 1) * np.power(10.0, np.random.randint(-10, 10))
        q = random_quantity_for_values(values)
        testing.assert_array_almost_equal(np.asarray(q.get_value()), values)


# comparison
def testComparisons(random_quantity_for_values) -> None:
    """Tests __lt__, __le__, __gt__, and __ge__ for scalar quantities."""
    for i in range(1, 100):
        # create a quantity with random values and check if the
        # get functions return the same values
        value = (2 * np.random.random() - 1) * np.power(10.0, np.random.randint(-10, 10))
        q1 = random_quantity_for_values(value)
        q2 = random_quantity_for_values(2 * np.abs(value))

        assert q2 > q1
        assert q2 >= q1
        assert q1 < q2
        assert q1 <= q2
        assert q1 <= q1
        assert q1 >= q1


def testEquality(random_quantity_for_values) -> None:
    """Tests __eq__ and __ne__ for scalar quantities."""
    for i in range(1, 100):
        # create a quantity with random values and check if the
        # get functions return the same values
        value = (2 * np.random.random() - 1) * np.power(10.0, np.random.randint(-10, 10))
        q1 = random_quantity_for_values(value)
        q2 = random_quantity_for_values(1.2 * np.abs(value))

        assert q1 != q2
        assert q2 != q1
        assert q1 == q1
        assert q2 == q2
        testing.assert_almost_equal(q1, value)
        testing.assert_almost_equal(q1, q1.get_value())


def testEqualityById(random_quantity):
    """Test __eq__ by ID."""
    q1 = random_quantity(1)
    q2 = Quantity(q1.get_value(), q1.get_min_value(), q1.get_max_value(), q1.get_unit())
    assert q1 == q2
    assert q1 is not q2
    assert q2 is not q1
    assert q1 is q1
    assert q2 is q2


def testNoInput():
    """Trying to instantiate without any parameters."""
    with pytest.raises(Exception):
        Quantity(np.random.random())
    with pytest.raises(Exception):
        Quantity()
    with pytest.raises(Exception):
        Quantity(np.random.random(), min_value=np.random.random(), max_value=None)
    with pytest.raises(Exception):
        Quantity(np.random.random(), min_value=None, max_value=np.random.random())


def testOutOfBounds():
    """Test out of bounds for scalar quantities."""
    num = Quantity(5, 3, 6)
    with pytest.raises(Exception):
        num.set_value(7)


def testRangeTooSmall():
    """Test if a too small range raises an exception."""
    r = np.random.random()

    with pytest.raises(Exception):
        Quantity(r, r, r)
    with pytest.raises(Exception):
        Quantity(r + 1e-10, r, r + 1e-20)

    num = Quantity(r, r - 1, r + 1)
    with pytest.raises(Exception):
        num.set_limits(r, r)
    with pytest.raises(Exception):
        num.set_value_and_limits(r, r, r)


def testArithmetic(five, three):
    """Test arithmetic special methods for scalar quantities."""
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
    """Test string conversion special methods for scalar quantities."""
    volts = Quantity(0.005, 0, 1, unit="V")
    assert str(volts) == "5 mV "
    resist = Quantity(2100, 0, 2500, unit="Ohm")
    assert str(resist) == "2.1 KOhm "
    amps = Quantity(125e6 * 2 * np.pi, 100e6, 1e9, unit="Hz", two_pi=True)
    assert str(amps) == "125 MHz x 2pi "


def testIsScalarOrVector(random_quantity):
    """Test whether quantity objects are scalars or vectors."""
    for i in range(20):
        q = random_quantity(1)
        assert q.is_scalar()
        assert not q.is_vector()

    for dim in range(2, 100):
        for i in range(20):
            q = random_quantity(dim)
            assert q.is_vector()
            assert not q.is_scalar()

            v = __generateRandomMatrix(dim)
            q2 = Quantity(v, v - 1, v + 1)
            assert not q2.is_vector()
            assert not q2.is_scalar()


def __generateRandomMatrix(N: int) -> np.ndarray:
    """Generate a random matrix of size `N` by `N`.

    Parameters
    ----------
    N : int
        Dimension of the matrix.

    Returns
    -------
    numpy.ndarray
        Returns a randomly generated `N` by `N` matrix.

    """
    magnitude: float = np.power(10.0, np.random.randint(-10, 10))
    return (2 * np.random.random((N, N)) - 1) * magnitude


def testRelations(three):
    """Test relation special methods for scalar quantities."""
    relation = Quantity.relational(three, lambda x: 2 * x)
    assert relation.dependent
    assert relation.get_value() == 6
    assert relation.get_min_value() == 0
    assert relation.get_max_value() == 15
    assert relation.get_unit() == ""
    assert relation.get_name() == "relation_of_"

    assert three.dependents == [relation]
    assert relation.dependencies == [three]

    three.set_value(4)
    assert relation.get_value() == 8
    with pytest.raises(ValueError):
        relation.set_value(7)

    copy = Quantity.relational_copy(relation)
    assert copy.dependent
    assert copy.get_value() == 8
    assert copy.get_min_value() == 0
    assert copy.get_max_value() == 15
    assert copy.get_unit() == ""
    assert copy.get_name() == "relation_of_"

    unit1 = Quantity(1, np.array(0), np.array(10), "Hz", "one")
    unit2 = Quantity(1, np.array(0), np.array(10), "s", "two")
    with pytest.raises(ValueError):
        _ = Quantity.relational([unit1, unit2], lambda x, y: x + y)
    with pytest.raises(ValueError):
        unit1.add_relation(unit2, lambda x: x)


def testPersistence(random_quantity):
    for N in range(1, 10):
        for i in range(20):
            q = random_quantity(N)
            data = q.to_dict()
            q2 = Quantity(0, -1, 1)
            q2.from_dict(data)

            # assert q == q2
            assert q.get_name() == q2.get_name()
            assert q.get_unit() == q2.get_unit()
            testing.assert_allclose(q.get_min_value(), q2.get_min_value())
            testing.assert_allclose(q.get_max_value(), q2.get_max_value())
            testing.assert_allclose(q.get_reduced_value(), q2.get_reduced_value())
            if N == 1:
                assert q2.is_scalar()
            else:
                assert q2.is_vector()
