"""Testing the optimisable maps."""

import numpy as np
import pytest
import random

from cthree.OptimisationMap import OptimisationMap
from test.TestOptimisable import TestOptimisable


@pytest.fixture
def randomOptimisables(randomQuantity):
    """Create random optimisables."""
    return [
        TestOptimisable(randomQuantity, np.random.randint(1, 10))
        for i in range(10)
    ]


@pytest.fixture
def optMapWithOptimisables(randomOptimisables):
    """Create a optimisation map with optimisables."""
    m = OptimisationMap()
    for i, optimisable in enumerate(randomOptimisables):
        m.add(optimisable)
    return m


def test_no_initial_parameters() -> None:
    """Test for initial parameters for a map.

    After initialisation, the map should not contain any
    optimisables or parameters.

    """
    m = OptimisationMap()
    assert len(m.getAllParameters()) == 0
    assert len(m.getOptimisables()) == 0


def test_adding_optimisables(randomOptimisables) -> None:
    """Test adding optimisables to a map.

    After adding an optimisable, it should be in the list.

    """
    m = OptimisationMap()
    for i, optimisable in enumerate(randomOptimisables):
        m.add(optimisable)
        assert len(m.getOptimisables()) == i + 1
        assert optimisable in m.getOptimisables()


def test_adding_all_parameters(randomOptimisables) -> None:
    """Test adding of all parameters.

    Adding an optimisable with all its parameters should
    increase to the number of parameters by the correct amount.

    """
    m = OptimisationMap()
    numParams = 0
    for i, optimisable in enumerate(randomOptimisables):
        numParams += len(optimisable.getParameters())

        m.add(optimisable)
        assert len(m.getAllParameters()) == numParams

        intersection = [
            p for p in optimisable.getParameters() if p in m.getAllParameters()
        ]
        assert len(intersection) == len(optimisable.getParameters())
        intersection2 = [
            p
            for p in optimisable.getParameters()
            if p in m.getParameters(optimisable)
        ]
        assert len(intersection2) == len(optimisable.getParameters())


def test_adding_some_parameters(randomOptimisables) -> None:
    """Test adding of some parameters.

    Adding an optimisable with some of its parameters should increase
    to the number of parameters by the correct amount.

    """
    m = OptimisationMap()
    numParams = 0
    for i, optimisable in enumerate(randomOptimisables):
        numAdded = (
            np.random.randint(1, len(optimisable.getParameters()))
            if len(optimisable.getParameters()) > 1
            else 1
        )
        parameters = random.sample(optimisable.getParameters(), numAdded)
        numParams += numAdded
        m.add(optimisable, parameters)

        allP = m.getAllParameters()
        assert len(allP) == numParams

        intersection = [
            p for p in optimisable.getParameters() if p in m.getAllParameters()
        ]
        assert len(intersection) == numAdded
        intersection2 = [
            p
            for p in optimisable.getParameters()
            if p in m.getParameters(optimisable)
        ]
        assert len(intersection2) == numAdded


def test_removing_optimisables(optMapWithOptimisables) -> None:
    """Test removing optimisables.

    After removing an optimisable, it should not be in the list anymore.

    """
    optimisables = optMapWithOptimisables.getOptimisables()

    for i, optimisable in enumerate(optimisables):
        optMapWithOptimisables.remove(optimisable)
        assert optimisable not in optMapWithOptimisables.getOptimisables()
        assert len(optMapWithOptimisables.getOptimisables()) == len(
            optimisables
        ) - (i + 1)


def test_removing_parameters(optMapWithOptimisables) -> None:
    """Test removing paramters.

    Removing an optimisable should decrease to the number of parameters
    by the correct amount.

    """
    optimisables = optMapWithOptimisables.getOptimisables()
    numParams = sum([len(o.getParameters()) for o in optimisables])

    for i, optimisable in enumerate(optimisables):
        numParams -= len(optimisable.getParameters())

        optMapWithOptimisables.remove(optimisable)
        assert len(optMapWithOptimisables.getAllParameters()) == numParams

        intersection = [
            p
            for p in optimisable.getParameters()
            if p in optMapWithOptimisables.getAllParameters()
        ]
        assert len(intersection) == 0
        with pytest.raises(Exception):
            optMapWithOptimisables.getParameters(optimisable)
