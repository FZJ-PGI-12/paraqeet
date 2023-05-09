from typing import List

import numpy as np

from cthree.Quantity import Quantity
from cthree.model.Model import Model
from cthree.propagation.Propagation import Propagation

import scipy

class ScipyExpm(Propagation):
    def __init__(self, model: Model):
        super().__init__(model)

    def getParameters(self) -> List[Quantity]:
        return []

    def propagate(self, time: np.ndarray):
        equationsOfMotion = self._model.getEquationOfMotion(time)
        return scipy.linalg.expm(equationsOfMotion)
