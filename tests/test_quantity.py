"""Test the quantity object."""

import numpy as np
import numpy.testing as testing
import pytest

from paraqeet.exceptions import IncompatibleQuantityException
from paraqeet.quantity import Quantity


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
    """If value, min, and max are arrays of length n, they should not be altered in the constructor."""
    for n in range(1, 100):
        values = np.random.rand(n)
        q = Quantity(values, min_value=0.9 * values, max_value=1.1 * values, unit="")
        assert len(q.get_value()) == n
        assert len(q.get_min_value()) == n
        assert len(q.get_max_value()) == n


def test_updates_bounds_length():
    """If value is of length N while min, and max are scalar, the bounds should be altered into arrays."""
    for n in range(1, 100):
        # In the constructor
        values = np.random.rand(n)
        q = Quantity(values, min_value=0.9 * np.min(values), max_value=1.1 * np.max(values), unit="")
        assert len(q.get_value()) == n

        min_value = q.get_min_value()
        assert len(min_value) == n
        testing.assert_allclose(min_value, min_value[0])

        max_value = q.get_max_value()
        assert len(max_value) == n
        testing.assert_allclose(max_value, max_value[0])

        # In set_limits
        q.set_limits(0.9 * np.min(values), 1.1 * np.max(values))
        min_value = q.get_min_value()
        assert len(min_value) == n
        testing.assert_allclose(min_value, min_value[0])

        max_value = q.get_max_value()
        assert len(max_value) == n
        testing.assert_allclose(max_value, max_value[0])


def test_fails_on_different_lengths(random_from_list):
    """If value, min, and max are arrays of different length, the quantity should raise an exception."""
    for n in range(1, 100):
        values = np.random.rand(n)
        # Test for 20 random shapes other than N if set_value, set_reduced_value, and set_value_and_limits fail
        for _ in range(20):
            min_shape = random_from_list(np.arange(100), n)
            min_values = 0.9 * np.random.rand(min_shape) * np.min(values)
            max_shape = random_from_list(np.arange(100), np.array([n, min_shape]))
            max_values = (1.0 + 0.1 * np.random.rand(max_shape)) * np.max(values)

            # In the constructor
            with pytest.raises(IncompatibleQuantityException):
                Quantity(values, min_value=min_values, max_value=max_values, unit="")

            # In set_limits
            q = Quantity(values, min_value=0.9 * values, max_value=1.1 * values, unit="")
            with pytest.raises(IncompatibleQuantityException):
                q.set_limits(min_values, max_values)
            testing.assert_array_equal(q.get_value(), values)
            testing.assert_almost_equal(q.get_min_value(), 0.9 * values)
            testing.assert_almost_equal(q.get_max_value(), 1.1 * values)

            # In set_value_and_limits
            with pytest.raises(IncompatibleQuantityException):
                q.set_value_and_limits(values, min_values, max_values)
            testing.assert_array_equal(q.get_value(), values)
            testing.assert_almost_equal(q.get_min_value(), 0.9 * values)
            testing.assert_almost_equal(q.get_max_value(), 1.1 * values)


# getter and setter
def test_get(random_quantity_for_values) -> None:
    """Test get_value, get_min_value, and get_max_value.

    For scalar quantities.

    """
    for n in range(1, 100):
        # create a quantity with random values and check if the
        # get functions return the same values
        values = (2 * np.random.random(n) - 1) * np.power(10.0, np.random.randint(-10, 10))
        q = random_quantity_for_values(values)
        testing.assert_allclose(q.get_value(), values)
        testing.assert_array_less(q.get_min_value(), q.get_value())
        testing.assert_array_less(q.get_value(), q.get_max_value())


def test_set(random_quantity_for_values, random_limits_for_quantity) -> None:
    """Test set_value, set_min_value, and set_max_value.

    For scalar quantities.

    """
    for n in range(1, 100):
        # create a random quantity with values that will be overwritten
        values = (2 * np.random.random(n) - 1) * np.power(10.0, np.random.randint(-10, 10))
        q = random_quantity_for_values(values)
        old_min = q.get_min_value()
        old_max = q.get_max_value()

        # set new values within the limits and check that the
        # limits stay unchanged
        new_values = np.random.random(n) * (old_max - old_min) + old_min
        q.set_value(new_values)
        testing.assert_allclose(q.get_value(), new_values)
        testing.assert_allclose(q.get_min_value(), old_min)
        testing.assert_allclose(q.get_max_value(), old_max)

        # set new limits and check that the values stay unchanged
        new_limits = random_limits_for_quantity(new_values)
        q.set_limits(*new_limits)
        testing.assert_allclose(q.get_min_value(), new_limits[0])
        testing.assert_allclose(q.get_max_value(), new_limits[1])
        testing.assert_array_less(q.get_min_value(), q.get_value())
        testing.assert_array_less(q.get_value(), q.get_max_value())


def test_set_fails_on_different_lengths(random_quantity_for_values, random_from_list):
    """Setting a new value should fail if the length of the new value is different than the old one."""
    for n in range(1, 100):
        # create a random quantity with values that shall be overwritten
        values = (2 * np.random.random(n) - 1) * np.power(10.0, np.random.randint(-10, 10))
        q = random_quantity_for_values(values)
        old_value = q.get_value()
        old_min = q.get_min_value()
        old_max = q.get_max_value()

        # Test for 20 random shapes other than N if set_value, set_reduced_value, and set_value_and_limits fail
        for _ in range(20):
            new_shape = random_from_list(np.arange(100), n)
            new_values = np.random.rand(new_shape)
            with pytest.raises(IncompatibleQuantityException):
                q.set_value(new_values)
            with pytest.raises(IncompatibleQuantityException):
                q.set_reduced_value(new_values)
            if n > 1:
                # If the bounds have a shape of N=1, set_value_and_limits would correctly assume that the bounds should
                # be transformed from scalar into array
                with pytest.raises(IncompatibleQuantityException):
                    q.set_value_and_limits(new_values, old_min, old_max)

        # Make sure that the quantity did not change
        testing.assert_array_equal(q.get_value(), old_value)
        testing.assert_array_equal(q.get_min_value(), old_min)
        testing.assert_array_equal(q.get_max_value(), old_max)


def test_get_item(random_quantity_for_values) -> None:
    """Test get item for scalar quantities."""
    for n in range(1, 100):
        values = (2 * np.random.random(n) - 1) * np.power(10.0, np.random.randint(-10, 10))
        q = random_quantity_for_values(values)
        for i in range(len(q)):
            testing.assert_allclose(q[i], values[i], rtol=1e-12)


def test_len(random_quantity_for_values) -> None:
    """Test the length value for scalar quantities."""
    for n in range(1, 100):
        values = (2 * np.random.random(n) - 1) * np.power(10.0, np.random.randint(-10, 10))
        q = random_quantity_for_values(values)
        testing.assert_almost_equal(len(q), n)


# conversion
def test_float(random_quantity_for_values) -> None:
    """Test float conversion for scalar quantities."""
    for _ in range(100):
        value = (2 * np.random.random(1) - 1) * np.power(10.0, np.random.randint(-10, 10))
        q = random_quantity_for_values(value)
        testing.assert_allclose(float(q), value, rtol=1e-12)


def test_to_array(random_quantity_for_values) -> None:
    """Test array conversion for scalar quantities."""
    for n in range(1, 100):
        values = (2 * np.random.random(n) - 1) * np.power(10.0, np.random.randint(-10, 10))
        q = random_quantity_for_values(values)
        testing.assert_allclose(np.asarray(q.get_value()), values, rtol=1e-12)


# comparison
def test_comparisons(random_quantity_for_values) -> None:
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


def test_equality(random_quantity_for_values) -> None:
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


def test_equality_by_id(random_quantity):
    """Test __eq__ by id."""
    q1 = random_quantity(1)
    q2 = Quantity(q1.get_value(), q1.get_min_value(), q1.get_max_value(), q1.get_unit())
    assert q1 == q2
    assert q1 is not q2
    assert q2 is not q1
    assert q1 is q1
    assert q2 is q2


def test_no_input():
    """Trying to instantiate without any parameters."""
    with pytest.raises(Exception):
        Quantity(np.random.random())
    with pytest.raises(Exception):
        Quantity()
    with pytest.raises(Exception):
        Quantity(np.random.random(), min_value=np.random.random(), max_value=None)
    with pytest.raises(Exception):
        Quantity(np.random.random(), min_value=None, max_value=np.random.random())


def test_out_of_bounds():
    """Test out of bounds for scalar quantities."""
    num = Quantity(5, 3, 6)
    with pytest.raises(Exception):
        num.set_value(7)


def test_range_too_small():
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


def test_arithmetic(five, three):
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


def test_str(five):
    """Test string conversion special methods for scalar quantities."""
    volts = Quantity(0.005, 0, 1, unit="V")
    assert str(volts) == "5 mV"
    resist = Quantity(2100, 0, 2500, unit="Ohm")
    assert str(resist) == "2.1 KOhm"
    amps = Quantity(125e6 * 2 * np.pi, 100e6, 1e9, unit="Hz", two_pi=True)
    assert str(amps) == "125 MHz x 2pi"
    bits = Quantity(np.array([256, 512]), 8, 1024, unit="Bits")
    assert str(bits) == "[256 Bits, 512 Bits]"


def test_is_scalar_or_vector(random_quantity):
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

            v = __generate_random_matrix(dim)
            q2 = Quantity(v, v - 1, v + 1)
            assert not q2.is_vector()
            assert not q2.is_scalar()


def __generate_random_matrix(n: int) -> np.ndarray:
    """Generate a random matrix of size `n` by `n`.

    Args:
        n: Dimension of the matrix.

    Returns:
        A randomly generated `N` by `N` matrix.
    """
    magnitude: float = np.power(10.0, np.random.randint(-10, 10))
    return (2 * np.random.random((n, n)) - 1) * magnitude


def test_relations(three):
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


def test_persistence(random_quantity):
    for n in range(1, 10):
        for _ in range(20):
            q = random_quantity(n)
            data = q.to_dict()
            q2 = Quantity(0, -1, 1)
            q2.from_dict(data)

            # assert q == q2
            assert q.get_name() == q2.get_name()
            assert q.get_unit() == q2.get_unit()
            testing.assert_allclose(q.get_min_value(), q2.get_min_value())
            testing.assert_allclose(q.get_max_value(), q2.get_max_value())
            testing.assert_allclose(q.get_reduced_value(), q2.get_reduced_value())
            if n == 1:
                assert q2.is_scalar()
            else:
                assert q2.is_vector()
