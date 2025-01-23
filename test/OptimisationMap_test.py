"""Testing the optimisable maps."""

import numpy as np
import pytest
import random

from cthree.OptimisationMap import OptimisationMap
from test.TestOptimisable import TestOptimisable


@pytest.fixture
def randomOptimisables(randomQuantity):
    """Create random optimisables."""
    return [TestOptimisable(randomQuantity, np.random.randint(2, 10)) for i in range(2, 10)]


@pytest.fixture
def optMapWithOptimisables(randomOptimisables):
    """Create a optimisation map with optimisables."""
    m = OptimisationMap()
    for i, optimisable in enumerate(randomOptimisables):
        m.add(optimisable)

        # Assign valid and unique names to the optimisable and its quantities
        optimisable.name = f"optimisable {i}"
        for j, quantity in enumerate(optimisable.get_parameters()):
            quantity.set_name(f"optimisable {i} - quantity {j}")
    return m


def test_no_initial_parameters() -> None:
    """Test for initial parameters for a map.

    After initialisation, the map should not contain any
    optimisables or parameters.

    """
    m = OptimisationMap()
    assert len(m.get_all_parameters()) == 0
    assert len(m.get_optimisables()) == 0


def test_adding_optimisables(randomOptimisables) -> None:
    """Test adding optimisables to a map.

    After adding an optimisable, it should be in the list.

    """
    m = OptimisationMap()
    for i, optimisable in enumerate(randomOptimisables):
        m.add(optimisable)
        assert len(m.get_optimisables()) == i + 1
        assert optimisable in m.get_optimisables()


def test_adding_all_parameters(randomOptimisables) -> None:
    """Test adding of all parameters.

    Adding an optimisable with all its parameters should
    increase to the number of parameters by the correct amount.

    """
    m = OptimisationMap()
    numParams = 0
    for i, optimisable in enumerate(randomOptimisables):
        numParams += len(optimisable.get_parameters())

        m.add(optimisable)
        assert len(m.get_all_parameters()) == numParams

        intersection = [p for p in optimisable.get_parameters() if p in m.get_all_parameters()]
        assert len(intersection) == len(optimisable.get_parameters())
        intersection2 = [p for p in optimisable.get_parameters() if p in m.get_parameters(optimisable)]
        assert len(intersection2) == len(optimisable.get_parameters())


def test_adding_some_parameters(randomOptimisables) -> None:
    """Test adding of some parameters.

    Adding an optimisable with some of its parameters should increase
    to the number of parameters by the correct amount.

    """
    m = OptimisationMap()
    numParams = 0
    for i, optimisable in enumerate(randomOptimisables):
        numAdded = (
            np.random.randint(1, len(optimisable.get_parameters())) if len(optimisable.get_parameters()) > 1 else 1
        )
        parameters = random.sample(optimisable.get_parameters(), numAdded)
        numParams += numAdded
        m.add(optimisable, parameters)

        allP = m.get_all_parameters()
        assert len(allP) == numParams

        intersection = [p for p in optimisable.get_parameters() if p in m.get_all_parameters()]
        assert len(intersection) == numAdded
        intersection2 = [p for p in optimisable.get_parameters() if p in m.get_parameters(optimisable)]
        assert len(intersection2) == numAdded


def test_removing_optimisables(optMapWithOptimisables) -> None:
    """Test removing optimisables.

    After removing an optimisable, it should not be in the list anymore.

    """
    optimisables = optMapWithOptimisables.get_optimisables()

    for i, optimisable in enumerate(optimisables):
        optMapWithOptimisables.remove(optimisable)
        assert optimisable not in optMapWithOptimisables.get_optimisables()
        assert len(optMapWithOptimisables.get_optimisables()) == len(optimisables) - (i + 1)


def test_removing_parameters(optMapWithOptimisables) -> None:
    """Test removing paramters.

    Removing an optimisable should decrease to the number of parameters
    by the correct amount.

    """
    optimisables = optMapWithOptimisables.get_optimisables()
    numParams = sum([len(o.get_parameters()) for o in optimisables])

    for i, optimisable in enumerate(optimisables):
        numParams -= len(optimisable.get_parameters())

        optMapWithOptimisables.remove(optimisable)
        assert len(optMapWithOptimisables.get_all_parameters()) == numParams

        intersection = [p for p in optimisable.get_parameters() if p in optMapWithOptimisables.get_all_parameters()]
        assert len(intersection) == 0
        with pytest.raises(Exception):
            optMapWithOptimisables.get_parameters(optimisable)


def test_exporting_fails(optMapWithOptimisables) -> None:
    """Tests if exporting fails if the optimisables or quantities do not have unique names."""
    optimisables = list(optMapWithOptimisables.get_optimisables())

    # Test bad names of optimisables
    notAllowed = [None, "", optimisables[1].name]
    for x in notAllowed:
        optimisables[0].name = x
        with pytest.raises(Exception):
            optMapWithOptimisables.to_dict()
    optimisables[0].name = "optimisable 0"

    # Test bad names of quantities
    quantities = optimisables[0].get_parameters()
    print("Changing: ", optimisables[0].name)
    notAllowed = [quantities[1].get_name()]
    for x in notAllowed:
        quantities[0].set_name(x)
        with pytest.raises(Exception):
            optMapWithOptimisables.to_dict()


def test_exporting(optMapWithOptimisables) -> None:
    """Tests if all optimisables and quantities are being exported."""
    optimisables = list(optMapWithOptimisables.get_optimisables())

    dictionary = optMapWithOptimisables.to_dict()
    assert len(dictionary) == len(optimisables)
    for optimisable in optimisables:
        assert optimisable.name in dictionary
        assert len(optimisable.get_parameters()) == len(dictionary[optimisable.name])
