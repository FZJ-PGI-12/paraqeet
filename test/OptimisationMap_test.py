import numpy as np
import pytest
import random

from cthree.OptimisationMap import OptimisationMap
from test.TestOptimisable import TestOptimisable


@pytest.fixture
def randomOptimisables(randomQuantity):
    return [TestOptimisable(randomQuantity, np.random.randint(1, 10)) for i in range(10)]


@pytest.fixture
def optMapWithOptimisables(randomOptimisables):
    m = OptimisationMap()
    for i, optimisable in enumerate(randomOptimisables):
        m.add(optimisable)
    return m


# After initialisation, the map should not contain any optimisables or parameters
def test_no_initial_parameters() -> None:
    m = OptimisationMap()
    assert len(m.getAllParameters()) == 0
    assert len(m.getOptimisables()) == 0


# After adding an optimisable, it should be in the list
def test_adding_optimisables(randomOptimisables) -> None:
    m = OptimisationMap()
    for i, optimisable in enumerate(randomOptimisables):
        m.add(optimisable)
        assert len(m.getOptimisables()) == i + 1
        assert optimisable in m.getOptimisables()


# Adding an optimisable with all its parameters should increase to the number of parameters by the correct amount
def test_adding_all_parameters(randomOptimisables) -> None:
    m = OptimisationMap()
    numParams = 0
    for i, optimisable in enumerate(randomOptimisables):
        numParams += len(optimisable.getParameters())

        m.add(optimisable)
        assert len(m.getAllParameters()) == numParams

        intersection = [p for p in optimisable.getParameters() if p in m.getAllParameters()]
        assert len(intersection) == len(optimisable.getParameters())
        intersection2 = [p for p in optimisable.getParameters() if p in m.getParameters(optimisable)]
        assert len(intersection2) == len(optimisable.getParameters())


# Adding an optimisable with some of its parameters should increase to the number of parameters by the correct amount
def test_adding_some_parameters(randomOptimisables) -> None:
    m = OptimisationMap()
    numParams = 0
    for i, optimisable in enumerate(randomOptimisables):
        numAdded = np.random.randint(1, len(optimisable.getParameters())) if len(optimisable.getParameters()) > 1 else 1
        parameters = random.sample(optimisable.getParameters(), numAdded)
        numParams += numAdded
        m.add(optimisable, parameters)

        allP = m.getAllParameters()
        assert len(allP) == numParams

        intersection = [p for p in optimisable.getParameters() if p in m.getAllParameters()]
        assert len(intersection) == numAdded
        intersection2 = [p for p in optimisable.getParameters() if p in m.getParameters(optimisable)]
        assert len(intersection2) == numAdded


# After removing an optimisable, it should not be in the list anymore
def test_removing_optimisables(optMapWithOptimisables) -> None:
    optimisables = optMapWithOptimisables.getOptimisables()

    for i, optimisable in enumerate(optimisables):
        optMapWithOptimisables.remove(optimisable)
        assert optimisable not in optMapWithOptimisables.getOptimisables()
        assert len(optMapWithOptimisables.getOptimisables()) == len(optimisables) - (i + 1)


# Removing an optimisable should decrease to the number of parameters by the correct amount
def test_removing_parameters(optMapWithOptimisables) -> None:
    optimisables = optMapWithOptimisables.getOptimisables()
    numParams = sum([len(o.getParameters()) for o in optimisables])

    for i, optimisable in enumerate(optimisables):
        numParams -= len(optimisable.getParameters())

        optMapWithOptimisables.remove(optimisable)
        assert len(optMapWithOptimisables.getAllParameters()) == numParams

        intersection = [p for p in optimisable.getParameters() if p in optMapWithOptimisables.getAllParameters()]
        assert len(intersection) == 0
        with pytest.raises(Exception):
            optMapWithOptimisables.getParameters(optimisable)
