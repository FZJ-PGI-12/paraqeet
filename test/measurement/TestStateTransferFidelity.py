import copy
from typing import Tuple

import numpy as np
import numpy.testing as testing

from cthree.Quantity import Quantity
from cthree.measurement.StateTransferFidelity import StateTransferFidelity

from test.RandomPropagation import RandomPropagation


# test that the fidelity for state vectors is always in the interval [0, 1)
def testLimitsVectors():
    for size in range(2, 30):
        propagation = RandomPropagation(size, False)
        target = propagation.propagate()
        measurement = StateTransferFidelity(propagation, target)

        for i in range(100):
            m = measurement.measure()
            assert 0.0 <= m <= 1.0


# test that the fidelity for density matrices is always in the interval [0, 1)
def testLimitsMatrices():
    for size in range(2, 30):
        propagation = RandomPropagation(size, True)
        target = propagation.propagate()
        measurement = StateTransferFidelity(propagation, target)
        for i in range(100):
            m = measurement.measure()
            assert 0.0 <= m <= 1.0


# test that F(v,v) = 1 for state vectors
def testVectorEquality():
    for size in range(2, 30):
        propagation = RandomPropagation(size, False, autoUpdate=False)

        for i in range(100):
            target = propagation.propagate()
            measurement = StateTransferFidelity(propagation, target)
            m = measurement.measure()
            np.testing.assert_almost_equal(m, 0.0)


# test that F(v,v) = 1 for density matrices
def testMatrixEquality():
    for size in range(2, 30):
        propagation = RandomPropagation(size, True, autoUpdate=False)

        for i in range(100):
            target = propagation.propagate()
            measurement = StateTransferFidelity(propagation, target)
            m = measurement.measure()
            np.testing.assert_almost_equal(m, 0.0)
