import numpy as np
from test.propagation.IdentityPropagation import IdentityPropagation
from cthree.measurement.MultiGoal import MultiGoal
from cthree.measurement.UnitaryFidelity import UnitaryFidelity


def test_multiGoal(randomUnitaryMatrix):
    meas = []
    for _ in range(np.random.randint(2, 10)):
        gate = randomUnitaryMatrix(np.random.randint(2, 4))
        propagation = IdentityPropagation()
        propagation.setInitialState(gate)
        meas.append(UnitaryFidelity(propagation, gate, np.array([1.0])))
    goal = MultiGoal(meas, weights=np.random.random(len(meas)))
    assert goal.measure() > 0
