import pytest
from cthree.propagation.RungeKutta import RungeKutta


@pytest.fixture
def rk(model):
    return RungeKutta(model)


def test_parameters(rk):
    assert rk.getParameters() == []


def test_propagation(rk, identity, ts):
    with pytest.raises(NotImplementedError):
        rk.propagate(identity, ts)
