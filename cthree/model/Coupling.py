from typing import List

from jax import vmap
import jax.numpy as jnp

from cthree.Optimisable import Optimisable
from cthree.Quantity import Quantity
from cthree.model.Hamiltonian import Hamiltonian

import jax

jax.config.update("jax_enable_x64", True)


class Coupling(Optimisable):
    """
    Represents the coupling of two or more subsystems in a composite Hamiltonian. This class implements longitudinal
    and transversal coupling with a constant scalar coefficient. The coefficient is the only optimisable parameter.
    Subclasses can alter the behavior by overriding the getMatrix function.

    :param subsystems: the coupled subsystems
    :param coefficient: either a constant coefficient as float or a callable that returns the coefficient for a
                        given time
    :param isLongitudinal: whether the coupling is longitudinal or transversal
    :param useRWA: if the transversal coupling should use the rotating-wave approximation or should include
                   double excitation terms
    """

    _subsystems: List[Hamiltonian]
    _coefficient: Quantity
    __isLongitudinal: bool

    def __init__(
        self,
        subsystems: List[Hamiltonian],
        coefficient: Quantity,
        isLongitudinal: bool,
        useRWA: bool = False,
    ):
        self._subsystems = subsystems
        self._coefficient = coefficient
        self.__isLongitudinal = isLongitudinal
        self.useRWA = useRWA
        self._totalDims = jnp.prod(
            jnp.array([s.dimension() for s in self.getSubsystems()])
        )

    def getParameters(self) -> List[Quantity]:
        return [self._coefficient]

    def getSubsystems(self) -> List[Hamiltonian]:
        """
        Returns all subsystems that are coupled by this term.
        """
        return self._subsystems

    def getMatricesOneTime(self, t: float) -> List[jnp.ndarray]:
        """
        Returns the matrix representation of the coupling for all subsystems. This assumes that the coupling factorises
        into terms for the subsystems, each of which is one element in the list. A composite Hamiltonian should take
        care of putting these terms into the correct position in the tensor space.
        """
        if self.useRWA:
            matrices = self.__couplingOperators()
            for i in range(len(matrices)):
                matrices[i][0] *= self._coefficient.getValue()
        else:
            matrices = self.__couplingOperators()
            matrices[0] *= self._coefficient.getValue()
        return matrices

    def getMatrices(self, t) -> List[jnp.ndarray]:
        """
        Returns the matrix representation of the coupling for all subsystems. This assumes that the coupling factorises
        into terms for the subsystems, each of which is one element in the list. A composite Hamiltonian should take
        care of putting these terms into the correct position in the tensor space.
        """
        return vmap(self.getMatricesOneTime)(t)

    def gradient(self, t) -> List[List[jnp.ndarray]]:
        """
        Returns the gradient of the matrix representation of the coupling for all subsystems. Each entry in the list
        is the gradient with respect to one parameter, factorised into subsystems.
        """
        if self._isOptimised(self._coefficient):
            grads = [self.__couplingOperators()]
        else:
            grads = jnp.empty((0, self._totalDims, self._totalDims))
        return grads

    def gradientOneTime(self, t) -> List[List[jnp.ndarray]]:
        """
        Returns the gradient of the matrix representation of the coupling for all subsystems. Each entry in the list
        is the gradient with respect to one parameter, factorised into subsystems.
        """
        return self.gradient(t)

    def __couplingOperators(self) -> List[jnp.ndarray]:
        """
        Returns the operators of the longitudinal or transversal coupling without coefficients.
        """
        if self.__isLongitudinal:
            # Number operator (a^\dagger a) for each subsystem
            return [
                jnp.diag(jnp.arange(0, s.dimension(), dtype=jnp.float64))
                for s in self._subsystems
            ]

        elif self.useRWA:
            # TODO - How to use RWA for more than 2 subsystems?

            if len(self.getSubsystems()) > 2:
                raise NotImplementedError("RWA is defined for 2 subsystems only")

            dimensions = [s.dimension() for s in self.getSubsystems()]
            annihilationOp = [
                jnp.sqrt(jnp.diag(jnp.arange(1, dim, dtype=jnp.float64), k=1))
                for dim in dimensions
            ]
            return [
                [annihilationOp[0], annihilationOp[1].T],
                [annihilationOp[0].T, annihilationOp[1]],
            ]

        else:
            # (a + a^\dagger) for each subsystem
            dimensions = [s.dimension() for s in self.getSubsystems()]
            annihilationOp = [
                jnp.sqrt(jnp.diag(jnp.arange(1, dim, dtype=jnp.float64), k=1))
                for dim in dimensions
            ]
            return [(a + a.T) for a in annihilationOp]
