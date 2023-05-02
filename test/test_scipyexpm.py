from numpy.testing import assert_almost_equal

import numpy as np
from cthree.propagation.ScipyExpm import ScipyExpm
from cthree.model.ClosedModel import ClosedModel


def test_identity() -> None:
    """
    Check that no input signal gives identity matrix as propagators.
    """
    num_points = 20
    T = np.linspace(0, 1, num_points)
    input_Hamiltonian = np.zeros((num_points, 10, 10))
    model = ClosedModel(input_Hamiltonian)
    propagation = ScipyExpm(model=model, T=T)
    propagators = propagation.propagate()

    identity = np.identity(10)

    assert_almost_equal(propagators, identity)
