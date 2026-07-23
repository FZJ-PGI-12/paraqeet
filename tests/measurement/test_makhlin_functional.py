"""Test the Makhlin Functional."""

import numpy as np
import pytest

from paraqeet.exceptions import ConfigurationException
from paraqeet.measurement.makhlin_functional import MakhlinFunctional
from tests.propagation.identity_propagation import IdentityPropagation
from tests.propagation.random_propagation import RandomPropagation

iswap = np.array([[1.0, 0, 0, 0], [0, 0, 1.0j, 0], [0, 1.0j, 0, 0], [0, 0, 0, 1.0]])
cnot = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0], [0, 0, 1.0, 0]])
swap = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])
sqrt_swap = np.array([[1.0, 0, 0, 0], [0, 1.0, 1.0, 0], [0, 1.0, 1.00, 0], [0, 0, 0, 1.0]])


def test_positivity():
    """Test the positivity.

    Whether distance is greater or equal to 0.

    """
    propagation = RandomPropagation(4, True)
    times = np.array([1.0])
    measurement = MakhlinFunctional(propagation.get_value)

    for _ in range(100):
        m = measurement.get_value(times)
        assert 0.0 <= m


def test_local_gates():
    """Test that local rotations have a distance of 2."""
    x = np.array(
        [
            [0, 1.0],
            [1.0, 0],
        ]
    )
    y = np.array(
        [
            [0, -1.0j],
            [1.0j, 0],
        ]
    )
    z = np.array([[1.0, 0], [0, -1.0]])
    gates1 = [np.kron(g, np.eye(2)) for g in [x, y, z]]
    gates2 = [np.kron(np.eye(2), g) for g in [x, y, z]]
    for gate in gates1 + gates2:
        propagation = IdentityPropagation()
        propagation.set_initial_state(gate)
        measurement = MakhlinFunctional(propagation.get_value)
        m = measurement.get_value(np.array([1.0]))
        np.testing.assert_almost_equal(m, 2.0)


def test_perfect_entanglers():
    """Test that perfect entangling gates have a distance of 0."""
    for gate in [iswap, cnot]:
        propagation = IdentityPropagation()
        propagation.set_initial_state(gate)
        measurement = MakhlinFunctional(propagation.get_value)
        m = measurement.get_value(np.array([1.0]))
        np.testing.assert_almost_equal(m, 0.0)


def test_invariants():
    """Test that special gates generate the expected Makhlin invariants."""
    expected_invariants = [
        [np.eye(4), [1, 0, 3]],
        [iswap, [0, 0, -1]],
        [cnot, [0, 0, 1]],
        [swap, [-1, 0, -3]],
    ]

    propagation = IdentityPropagation()
    for gate, invariants in expected_invariants:
        propagation.set_initial_state(gate)
        measurement = MakhlinFunctional(propagation.get_value, np.array(invariants))
        m = measurement.get_value(np.array([1.0]))
        np.testing.assert_almost_equal(m, 0.0)


def test_incompatible_shape():
    """Test that all propagators which are not 4D raise an exception.

    Raises
    ------
    Exception
        Raise an error if all propagators are not 4D.

    """
    incompatible_dimensions = np.delete(np.arange(2, 30), 2)
    for dim in incompatible_dimensions:
        propagation = RandomPropagation(dim, True)
        measurement = MakhlinFunctional(propagation.get_value)
        with pytest.raises(Exception):
            _ = measurement.get_value(np.array([1.0]))


def test_measurement_needs_time():
    propagation = RandomPropagation(4, True)
    measurement = MakhlinFunctional(propagation.get_value, None)
    with pytest.raises(ConfigurationException):
        measurement.get_value(times=None)
