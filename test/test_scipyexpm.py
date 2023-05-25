from numpy.testing import assert_almost_equal

import numpy as np

from cthree.propagation.ScipyExpm import ScipyExpm


def test_parameters(model):
    propagation = ScipyExpm(model=model)
    assert propagation.getParameters() == []


def test_identity(model, ts, identity) -> None:
    """
    Check that no input signal gives identity matrix as propagators.
    """
    propagation = ScipyExpm(model=model)
    propagators = propagation.propagate(ts)

    for prop in propagators:
        assert_almost_equal(np.abs(prop), identity)
