from numpy.testing import assert_almost_equal
from typing import List
import numpy as np
from cthree.propagation.ScipyExpm import ScipyExpm
from cthree.model.ClosedModel import ClosedModel
from cthree.signal.Device import ZeroTone
from cthree.signal.SimpleGenerator import CosGenerator
from cthree.model.Hamiltonian import Hamiltonian
from cthree.model.Model import Model
from cthree.Quantity import Quantity


LEN_SIG = 20
DIMS = 10


ts = np.linspace(0, 1e-9, LEN_SIG)
dummy_hamiltonian = np.zeros((LEN_SIG, DIMS, DIMS))


class dummy_model(Model):
    def __init__(self, hamiltonian: Hamiltonian):
        super().__init__(hamiltonian)

    def getParameters(self) -> List[Quantity]:
        pass

    def getEquationOfMotion(self, t: np.ndarray) -> np.ndarray:
        return -1.0j * self._hamiltonian


def test_identity() -> None:
    """
    Check that no input signal gives identity matrix as propagators.
    """
    model = dummy_model(dummy_hamiltonian)
    propagation = ScipyExpm(model=model, T=ts)
    propagators = propagation.propagate()

    identity = np.identity(DIMS)

    for prop in propagators:
        assert_almost_equal(np.abs(prop), identity)
