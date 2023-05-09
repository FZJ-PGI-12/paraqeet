from numpy.testing import assert_almost_equal
from typing import List
import numpy as np
from cthree.propagation.ScipyExpm import ScipyExpm
from cthree.model.Hamiltonian import Hamiltonian
from cthree.model.Model import Model
from cthree.Quantity import Quantity
from cthree.signal.Generator import Generator

LEN_SIG = 20
DIMS = 10


ts = np.linspace(0, 1e-9, LEN_SIG)


class DummyHamiltonian(Hamiltonian):
    def __init__(self):
        super().__init__([], None, None, Generator())

    def getMatrix(self, t: np.ndarray) -> np.ndarray:
        return np.zeros((LEN_SIG, DIMS, DIMS))


class dummy_model(Model):
    def __init__(self, hamiltonian: Hamiltonian):
        super().__init__(hamiltonian)

    def getParameters(self) -> List[Quantity]:
        pass

    def getEquationOfMotion(self, t: np.ndarray) -> np.ndarray:
        return -1.0j * self._hamiltonian.getMatrix(t) * (t[1] - t[0])


def test_identity() -> None:
    """
    Check that no input signal gives identity matrix as propagators.
    """
    model = dummy_model(DummyHamiltonian())
    propagation = ScipyExpm(model=model)
    propagators = propagation.propagate(ts)

    identity = np.identity(DIMS)

    for prop in propagators:
        assert_almost_equal(np.abs(prop), identity)
