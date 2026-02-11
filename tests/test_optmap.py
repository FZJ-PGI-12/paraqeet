"""Testing the optimization map."""

import random

import numpy as np
import pytest

from paraqeet.exceptions import ConfigurationException, SerializationException
from paraqeet.optimization_map import OptimizationMap
from paraqeet.signal.envelopes import ConstantEnvelope, FlatTopGaussianEnvelope
from paraqeet.signal.iq_mixer import IQMixer
from tests.test_optimizable import DummyOptimizable

TONE = ConstantEnvelope()
GEN = IQMixer(envelopes=[TONE])
PARAMS = TONE.get_parameters()


@pytest.fixture
def optmap():
    return OptimizationMap()


@pytest.fixture
def dummy_optimizable(random_quantity):
    def _method(num_params: int) -> DummyOptimizable:
        return DummyOptimizable(random_quantity, num_params)

    return _method


@pytest.fixture
def random_optimizables(random_quantity):
    """Create random optimizables."""
    return [DummyOptimizable(random_quantity, np.random.randint(2, 10)) for i in range(2, 10)]


@pytest.fixture
def opt_map_with_optimizables(random_optimizables):
    """Create a optimization map with optimizables."""
    optmap = OptimizationMap()
    for i, optimizable in enumerate(random_optimizables):
        optmap.add(optimizable)

        # Assign valid and unique names to the optimizable and its quantities
        optimizable.name = f"optimizable {i}"
        for j, quantity in enumerate(optimizable.get_parameters()):
            quantity.set_name(f"optimizable {i} - quantity {j}")
    return optmap


def test_get_parameters(optmap) -> None:
    """Get the test parameters from the optimization map."""
    optmap.add(TONE, PARAMS)
    assert len(optmap.get_all_parameters()) == len(PARAMS)


def test_parameters_overwrite(optmap) -> None:
    """Override parameters from the optimization map."""
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


def test_properties(optmap, dummy_optimizable) -> None:
    optmap.add(dummy_optimizable(1))
    assert type(optmap.__str__()) is str
    assert type(optmap.__repr__()) is str


def test_optimizables_are_added(optmap, dummy_optimizable) -> None:
    opt1 = dummy_optimizable(1)
    optmap.add(opt1)
    assert opt1 in optmap.get_optimizables()
    assert optmap.get_parameters(opt1) is not None
    assert len(optmap.get_all_parameters()) == 1
    optmap.remove(opt1)

    opt2 = dummy_optimizable(0)
    optmap.add(opt2)
    assert opt2 not in optmap.get_optimizables()
    assert len(optmap.get_optimizables()) == 0
    with pytest.raises(Exception):
        optmap.get_parameters(opt2)
    assert len(optmap.get_all_parameters()) == 0


def test_access_to_all_optimizables_parameters(optmap, dummy_optimizable) -> None:
    """Test for accessing all the otpimisable parameters in the optmap from
    the Optimizable objects.
    """
    opt1 = dummy_optimizable(1)
    opt2 = dummy_optimizable(2)
    optmap.add(opt1)
    optmap.add(opt2)
    optmap.register_params_with_optimizables()
    assert len(opt1.optimizable_parameters) < len(optmap.get_all_parameters())
    assert len(opt2.optimizable_parameters) < len(optmap.get_all_parameters())
    assert optmap.get_all_parameters() == opt1.all_optimizable_parameters
    assert optmap.get_all_parameters() == opt2.all_optimizable_parameters


def test_no_initial_parameters() -> None:
    """Test for initial parameters for a map.

    After initialisation, the map should not contain any
    optimizables or parameters.

    """
    optmap = OptimizationMap()
    assert len(optmap.get_all_parameters()) == 0
    assert len(optmap.get_optimizables()) == 0


def test_adding_optimizables(random_optimizables) -> None:
    """Test adding optimizables to a map.

    After adding an optimizable, it should be in the list.

    """
    optmap = OptimizationMap()
    for i, optimizable in enumerate(random_optimizables):
        optmap.add(optimizable)
        assert len(optmap.get_optimizables()) == i + 1
        assert optimizable in optmap.get_optimizables()


def test_adding_all_parameters(random_optimizables) -> None:
    """Test adding of all parameters.

    Adding an optimizable with all its parameters should
    increase to the number of parameters by the correct amount.

    """
    optmap = OptimizationMap()
    num_params = 0
    for _, optimizable in enumerate(random_optimizables):
        num_params += len(optimizable.get_parameters())

        optmap.add(optimizable)
        assert len(optmap.get_all_parameters()) == num_params

        intersection = [p for p in optimizable.get_parameters() if p in optmap.get_all_parameters()]
        assert len(intersection) == len(optimizable.get_parameters())
        parameters = optmap.get_parameters(optimizable)
        if parameters is not None:
            intersection2 = [p for p in optimizable.get_parameters() if p in parameters]
        else:
            raise ConfigurationException(
                f"{optimizable}.get_parameters() returns None. No quantities specified in {optimizable}."
            )
        assert len(intersection2) == len(optimizable.get_parameters())


def test_adding_some_parameters(random_optimizables) -> None:
    """Test adding of some parameters.

    Adding an optimizable with some of its parameters should increase
    to the number of parameters by the correct amount.

    """
    optmap = OptimizationMap()
    num_params = 0
    for _, optimizable in enumerate(random_optimizables):
        num_added = (
            np.random.randint(1, len(optimizable.get_parameters())) if len(optimizable.get_parameters()) > 1 else 1
        )
        parameters = random.sample(optimizable.get_parameters(), num_added)
        num_params += num_added
        optmap.add(optimizable, parameters)

        all_p = optmap.get_all_parameters()
        assert len(all_p) == num_params

        intersection = [p for p in optimizable.get_parameters() if p in optmap.get_all_parameters()]
        assert len(intersection) == num_added
        new_parameters = optmap.get_parameters(optimizable)
        if new_parameters is not None:
            intersection2 = [p for p in optimizable.get_parameters() if p in new_parameters]
        else:
            raise ConfigurationException(
                f"{optimizable}.get_parameters() returns None. No quantities specified in {optimizable}."
            )
        assert len(intersection2) == num_added


def test_removing_optimizables(opt_map_with_optimizables) -> None:
    """Test removing optimizables.

    After removing an optimizable, it should not be in the list anymore.

    """
    optimizables = opt_map_with_optimizables.get_optimizables()

    for i, optimizable in enumerate(optimizables):
        opt_map_with_optimizables.remove(optimizable)
        assert optimizable not in opt_map_with_optimizables.get_optimizables()
        assert len(opt_map_with_optimizables.get_optimizables()) == len(optimizables) - (i + 1)


def test_removing_parameters(opt_map_with_optimizables) -> None:
    """Test removing paramters.

    Removing an optimizable should decrease to the number of parameters
    by the correct amount.

    """
    optimizables = opt_map_with_optimizables.get_optimizables()
    num_params = sum([len(o.get_parameters()) for o in optimizables])

    for _, optimizable in enumerate(optimizables):
        num_params -= len(optimizable.get_parameters())

        opt_map_with_optimizables.remove(optimizable)
        assert len(opt_map_with_optimizables.get_all_parameters()) == num_params

        intersection = [p for p in optimizable.get_parameters() if p in opt_map_with_optimizables.get_all_parameters()]
        assert len(intersection) == 0
        with pytest.raises(Exception):
            opt_map_with_optimizables.get_parameters(optimizable)


def test_exporting_fails(opt_map_with_optimizables) -> None:
    """Tests if exporting fails if the optimizables or quantities do not have unique names."""
    optimizables = list(opt_map_with_optimizables.get_optimizables())

    # Test bad names of optimizables
    not_allowed = [None, "", optimizables[1].name]
    for x in not_allowed:
        optimizables[0].name = x
        with pytest.raises(Exception):
            opt_map_with_optimizables.to_dict()
    optimizables[0].name = "optimizable 0"

    # Test bad names of quantities
    quantities = optimizables[0].get_parameters()
    print("Changing: ", optimizables[0].name)
    not_allowed = [quantities[1].get_name()]
    for x in not_allowed:
        quantities[0].set_name(x)
        with pytest.raises(Exception):
            opt_map_with_optimizables.to_dict()


def test_exporting(opt_map_with_optimizables, dummy_optimizable) -> None:
    """Tests if all optimizables and quantities are being exported."""
    optimizables = list(opt_map_with_optimizables.get_optimizables())

    dictionary = opt_map_with_optimizables.to_dict()
    assert len(dictionary) == len(optimizables)
    for optimizable in optimizables:
        assert optimizable.name in dictionary
        assert len(optimizable.get_parameters()) == len(dictionary[optimizable.name])

    # Test if re-importing works
    opt_map_with_optimizables.from_dict(dictionary)
    assert list(opt_map_with_optimizables.get_optimizables()) == optimizables

    # Test if importing fails if an optimizable does not yet exist in the optmap
    dictionary["new-optimizable"] = dummy_optimizable(1)
    with pytest.raises(SerializationException):
        opt_map_with_optimizables.from_dict(dictionary)
