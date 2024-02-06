import numpy as np
from typing import List, Tuple

from cthree.propagation.ScipyExpm import ScipyExpm

from scipy.linalg import block_diag


class ScipyExpmGOAT(ScipyExpm):
    """
    Solve the equation of motion by piecewise exponentation with the scipy package.
    """

    _res: float
    _initialState: np.ndarray = None

    def gradient(self, time: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the GOAT equation for the gradient vector.

        Parameters
        ----------
        time : np.ndarray
            array of timesteps

        Returns
        -------
        np.ndarray
            first dimension is time, second dimension is the parameter
        """
        eom = self._model.getMatrixEOM

        n_params = len(self._model.gradient(0))

        psi = [self._initialState]
        dpsis = [[np.zeros_like(self._initialState)] * n_params]

        for ti in range(1, len(time)):
            times, dt = self._constructTimes(time, ti)
            superState = [psi[-1]]
            superState.extend(dpsis[-1])
            psis_t = np.concatenate(superState)
            for t in times:
                # Sampling at the center of the interval.
                hamiltonian = eom(np.reshape(t, (-1, 1)) + dt / 2)
                dim = hamiltonian.shape[0]

                # Get the gradients of the MatrixEOM
                EOM_grad = self._model.gradient(t + dt / 2)

                line = [hamiltonian]
                line.extend([np.zeros_like(hamiltonian)] * n_params)
                goat_ham_list = [line]

                for ii, dH_dp in enumerate(EOM_grad, start=1):
                    line = [dH_dp]
                    line.extend([np.zeros_like(hamiltonian)] * (ii-1))
                    line.append(hamiltonian)
                    line.extend([np.zeros_like(hamiltonian)] * (n_params - ii))
                    goat_ham_list.append(line)

                psis_t = self._propagatePsi(np.block(goat_ham_list) * dt, psis_t)
            psi.append(psis_t[0:dim])
            dpsis.append(
                np.array([psis_t[dim * ii : dim * (ii + 1)] for ii in range(1, n_params + 1)])
            )
        return np.array(psi), np.array(dpsis)
