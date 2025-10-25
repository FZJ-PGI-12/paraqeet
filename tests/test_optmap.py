"""Testing the optimisation map."""

import numpy as np
import pytest
import random

from paraqeet.exceptions import ConfigurationException, SerialisationException
from paraqeet.optimisation_map import OptimisationMap

from paraqeet.signal.iq_mixer import IQMixer
from paraqeet.signal.envelopes import ConstantEnvelope, FlatTopGaussianEnvelope
from tests.test_optimisable import DummyOptimisable

TONE = ConstantEnvelope()
GEN = IQMixer(envelopes=[TONE])
PARAMS = TONE.get_parameters()


@pytest.fixture
def optmap():
    return OptimisationMap()


@pytest.fixture
def dummy_optimisable(random_quantity):
    def _method(num_params: int) -> DummyOptimisable:
        return DummyOptimisable(random_quantity, num_params)

    return _method


@pytest.fixture
def random_optimisables(random_quantity):
    """Create random optimisables."""
    return [DummyOptimisable(random_quantity, np.random.randint(2, 10)) for i in range(2, 10)]


@pytest.fixture
def opt_map_with_optimisables(random_optimisables):
    """Create a optimisation map with optimisables."""
    optmap = OptimisationMap()
    for i, optimisable in enumerate(random_optimisables):
        optmap.add(optimisable)

        # Assign valid and unique names to the optimisable and its quantities
        optimisable.name = f"optimisable {i}"
        for j, quantity in enumerate(optimisable.get_parameters()):
            quantity.set_name(f"optimisable {i} - quantity {j}")
    return optmap


def test_get_parameters(optmap) -> None:
    """Get the test parameters from the optimisation map."""
    optmap.add(TONE, PARAMS)
    assert len(optmap.get_all_parameters()) == len(PARAMS)


def test_parameters_overwrite(optmap) -> None:
    """Override parameters from the optimisation map."""
    optmap.add(TONE, [PARAMS[1]])
    assert optmap.get_all_parameters() == [PARAMS[1]]


def test_filter(optmap) -> None:
    """Test for filtered parameters."""
    tone2 = FlatTopGaussianEnvelope()
    optmap.add(tone2)

    def HzFilter(par):
        return par.get_unit() == "Hz"

    # Manual filtering
    pars = optmap.get_all_parameters()
    filterd = []
    for par in pars:
        if HzFilter(par):
            filterd.append(par)

    # Builtin filter
    optmap.filter_parameters(HzFilter)
    pars = optmap.get_all_parameters()
    assert pars == filterd


def test_properties(optmap, dummy_optimisable) -> None:
    optmap.add(dummy_optimisable(1))
    assert type(optmap.__str__()) is str
    assert type(optmap.__repr__()) is str


def test_optimisables_are_added(optmap, dummy_optimisable) -> None:
    opt1 = dummy_optimisable(1)
    optmap.add(opt1)
    assert opt1 in optmap.get_optimisables()
    assert optmap.get_parameters(opt1) is not None
    assert len(optmap.get_all_parameters()) == 1
    optmap.remove(opt1)

    opt2 = dummy_optimisable(0)
    optmap.add(opt2)
    assert opt2 not in optmap.get_optimisables()
    assert len(optmap.get_optimisables()) == 0
    with pytest.raises(Exception):
        optmap.get_parameters(opt2)
    assert len(optmap.get_all_parameters()) == 0


def test_access_to_all_optimisables_parameters(optmap, dummy_optimisable) -> None:
    """Test for accessing all the otpimisable parameters in the optmap from
    the Optimisable objects.
    """
    opt1 = dummy_optimisable(1)
    opt2 = dummy_optimisable(2)
    optmap.add(opt1)
    optmap.add(opt2)
    optmap.register_params_with_optimisables()
    assert len(opt1.optimisable_parameters) < len(optmap.get_all_parameters())
    assert len(opt2.optimisable_parameters) < len(optmap.get_all_parameters())
    assert optmap.get_all_parameters() == opt1.all_optimisable_parameters
    assert optmap.get_all_parameters() == opt2.all_optimisable_parameters


def test_no_initial_parameters() -> None:
    """Test for initial parameters for a map.

    After initialisation, the map should not contain any
    optimisables or parameters.

    """
    optmap = OptimisationMap()
    assert len(optmap.get_all_parameters()) == 0
    assert len(optmap.get_optimisables()) == 0


def test_adding_optimisables(random_optimisables) -> None:
    """Test adding optimisables to a map.

    After adding an optimisable, it should be in the list.

    """
    optmap = OptimisationMap()
    for i, optimisable in enumerate(random_optimisables):
        optmap.add(optimisable)
        assert len(optmap.get_optimisables()) == i + 1
        assert optimisable in optmap.get_optimisables()


def test_adding_all_parameters(random_optimisables) -> None:
    """Test adding of all parameters.

    Adding an optimisable with all its parameters should
    increase to the number of parameters by the correct amount.

    """
    optmap = OptimisationMap()
    num_params = 0
    for _, optimisable in enumerate(random_optimisables):
        num_params += len(optimisable.get_parameters())

        optmap.add(optimisable)
        assert len(optmap.get_all_parameters()) == num_params

        intersection = [p for p in optimisable.get_parameters() if p in optmap.get_all_parameters()]
        assert len(intersection) == len(optimisable.get_parameters())
        parameters = optmap.get_parameters(optimisable)
        if parameters is not None:
            intersection2 = [p for p in optimisable.get_parameters() if p in parameters]
        else:
            raise ConfigurationException(
                f"{optimisable}.get_parameters() returns None. No quantities specified in {optimisable}."
            )
        assert len(intersection2) == len(optimisable.get_parameters())


def test_adding_some_parameters(random_optimisables) -> None:
    """Test adding of some parameters.

    Adding an optimisable with some of its parameters should increase
    to the number of parameters by the correct amount.

    """
    optmap = OptimisationMap()
    num_params = 0
    for _, optimisable in enumerate(random_optimisables):
        num_added = (
            np.random.randint(1, len(optimisable.get_parameters())) if len(optimisable.get_parameters()) > 1 else 1
        )
        parameters = random.sample(optimisable.get_parameters(), num_added)
        num_params += num_added
        optmap.add(optimisable, parameters)

        all_p = optmap.get_all_parameters()
        assert len(all_p) == num_params

        intersection = [p for p in optimisable.get_parameters() if p in optmap.get_all_parameters()]
        assert len(intersection) == num_added
        new_parameters = optmap.get_parameters(optimisable)
        if new_parameters is not None:
            intersection2 = [p for p in optimisable.get_parameters() if p in new_parameters]
        else:
            raise ConfigurationException(
                f"{optimisable}.get_parameters() returns None. No quantities specified in {optimisable}."
            )
        assert len(intersection2) == num_added


def test_removing_optimisables(opt_map_with_optimisables) -> None:
    """Test removing optimisables.

    After removing an optimisable, it should not be in the list anymore.

    """
    optimisables = opt_map_with_optimisables.get_optimisables()

    for i, optimisable in enumerate(optimisables):
        opt_map_with_optimisables.remove(optimisable)
        assert optimisable not in opt_map_with_optimisables.get_optimisables()
        assert len(opt_map_with_optimisables.get_optimisables()) == len(optimisables) - (i + 1)


def test_removing_parameters(opt_map_with_optimisables) -> None:
    """Test removing paramters.

    Removing an optimisable should decrease to the number of parameters
    by the correct amount.

    """
    optimisables = opt_map_with_optimisables.get_optimisables()
    num_params = sum([len(o.get_parameters()) for o in optimisables])

    for _, optimisable in enumerate(optimisables):
        num_params -= len(optimisable.get_parameters())

        opt_map_with_optimisables.remove(optimisable)
        assert len(opt_map_with_optimisables.get_all_parameters()) == num_params

        intersection = [p for p in optimisable.get_parameters() if p in opt_map_with_optimisables.get_all_parameters()]
        assert len(intersection) == 0
        with pytest.raises(Exception):
            opt_map_with_optimisables.get_parameters(optimisable)


def test_exporting_fails(opt_map_with_optimisables) -> None:
    """Tests if exporting fails if the optimisables or quantities do not have unique names."""
    optimisables = list(opt_map_with_optimisables.get_optimisables())

    # Test bad names of optimisables
    not_allowed = [None, "", optimisables[1].name]
    for x in not_allowed:
        optimisables[0].name = x
        with pytest.raises(Exception):
            opt_map_with_optimisables.to_dict()
    optimisables[0].name = "optimisable 0"

    # Test bad names of quantities
    quantities = optimisables[0].get_parameters()
    print("Changing: ", optimisables[0].name)
    not_allowed = [quantities[1].get_name()]
    for x in not_allowed:
        quantities[0].set_name(x)
        with pytest.raises(Exception):
            opt_map_with_optimisables.to_dict()


def test_exporting(opt_map_with_optimisables, dummy_optimisable) -> None:
    """Tests if all optimisables and quantities are being exported."""
    optimisables = list(opt_map_with_optimisables.get_optimisables())

    dictionary = opt_map_with_optimisables.to_dict()
    assert len(dictionary) == len(optimisables)
    for optimisable in optimisables:
        assert optimisable.name in dictionary
        assert len(optimisable.get_parameters()) == len(dictionary[optimisable.name])

    # Test if re-importing works
    opt_map_with_optimisables.from_dict(dictionary)
    assert list(opt_map_with_optimisables.get_optimisables()) == optimisables

    # Test if importing fails if an optimisable does not yet exist in the optmap
    dictionary["new-optimisable"] = dummy_optimisable(1)
    with pytest.raises(SerialisationException):
        opt_map_with_optimisables.from_dict(dictionary)
