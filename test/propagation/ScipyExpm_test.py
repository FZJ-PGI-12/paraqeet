import numpy as np

from cthree.propagation.ScipyExpm import ScipyExpm


def test_parameters(model):
    propagation = ScipyExpm(model=model, res=3)
    assert propagation.getParameters() == []


def test_resolution(model):
    for i in range(10):
        propagation = ScipyExpm(model=model, res=3)
        resolution = np.random.randint(1, 1000)
        propagation.setResolution(resolution)
        assert propagation.getResolution() == resolution
