import numpy as np

from cthree.propagation.ScipyExpm import ScipyExpm

import scipy


class ScipyExpmGOAT(ScipyExpm):
    """
    Solve the equation of motion by piecewise exponentation with the scipy package.
    """

    _res: float
    _initialState: np.ndarray = None

    def gradient(self, time: np.ndarray):
        """
        Solve the GOAT equation for the gradient vector
        """
        eom = self._model.getMatrixEOM

        n_params = 2

        psi = [self._initialState]
        dpsis = [[np.zeros_like(self._initialState)] * n_params]

        for ti in range(1, len(time)):
            t0 = time[ti - 1]
            t1 = time[ti]
            steps = int(np.ceil((t1 - t0) * self.__res))
            times = np.linspace(t0, t1, steps, endpoint=False)
            if steps < 2:
                dt = t1 - t0
            else:
                dt = times[1] - times[0]
            superState = [psi[-1]]
            superState.extend(dpsis[-1])
            psis_t = np.concatenate(superState)
            for t in times:
                # Sampling at the center of the interval.
                this_h = eom(np.reshape(t, (-1, 1)) + dt / 2)
                dim = this_h.shape[0]

                # Get the gradients of the MatrixEOM
                EOM_grad = self._model.gradient(t + dt / 2)

                # Initialize the GOAT H with the diagonal
                h_list = [this_h] * (n_params + 1)
                goat_ham = scipy.linalg.block_diag(*h_list)

                # Add the first column of derivatives
                for ii, dH_dp in enumerate(EOM_grad):
                    goat_ham[
                        np.ix_([dim * (ii + 1), dim * (ii + 2) - 1], [0, dim - 1])
                    ] = dH_dp

                psis_t = scipy.linalg.expm(goat_ham * dt) @ psis_t
            psi.append(psis_t[0:dim])
            dpsis.append(
                [psis_t[dim * ii : dim * (ii + 1)] for ii in range(1, n_params + 1)]
            )
        return dpsis
