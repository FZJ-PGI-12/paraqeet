"""Class definition of the 7th-order Verner ODE solver for open quantum system."""

from functools import partial

import numpy as np
import jax.numpy as jnp

from cthree.exceptions import ConfigurationException
from cthree.quantity import Quantity
from cthree.model.equation_of_motion import EquationOfMotion
from cthree.propagation.state_propagation import StatePropagation

import jax
from jax import Array, jit
from jax.lax import scan, dynamic_slice_in_dim

jax.config.update("jax_enable_x64", True)


class Vern7(StatePropagation):
    """
    Propagate state by solving the Lindblad master equation by using ODE solver.

    Implements Vern7 ODE Solver algorithm non adaptive version.
    It has a fixed step size right now.
    """

    _res: float
    _initial_state: np.ndarray | None = None

    def __init__(self, model: EquationOfMotion, res: float):
        """
        Parameters
        ----------
        model: Model
            Model
        res: float
            Resolution at which to sample the EOM
        """
        super().__init__(model)
        self.resolution = res

    @property
    def resolution(self) -> float:
        """Get the resolution of the system."""
        return self._res

    @resolution.setter
    def resolution(self, res: float):
        """Set the resolution of the propagation."""
        self._res = res

    def get_parameters(self) -> list[Quantity]:
        """
        Method has no optimizable parameters.

        Returns
        -------
            Empty list
        """
        return []

    @staticmethod
    def _commutator(A: jnp.ndarray, B: jnp.ndarray):
        return jnp.matmul(A, B) - jnp.matmul(B, A)

    @staticmethod
    def _anti_commutator(A: jnp.ndarray, B: jnp.ndarray):
        return jnp.matmul(A, B) + jnp.matmul(B, A)

    @staticmethod
    def _dagger(op: jnp.ndarray):
        return op.conj().T

    def _construct_times(self, time, ti):
        """Construct one-dimensional vector of time."""
        t0 = time[ti - 1]
        t1 = time[ti]
        steps = int(np.ceil((t1 - t0) * self._res))
        times = np.linspace(t0, t1, steps, endpoint=False)
        if steps < 2:
            dt = t1 - t0
        else:
            dt = times[1] - times[0]
        return times, dt

    @staticmethod
    def _interpolate_time(times, dt):
        times_interp = jnp.concatenate(
            [
                times,
                times + (1 / 200) * dt,
                times + (49 / 450) * dt,
                times + (49 / 300) * dt,
                times + (911 / 2000) * dt,
                times + (3480084980 / 5709648941) * dt,
                times + (221 / 250) * dt,
                times + (37 / 40) * dt,
                times + dt,
            ],
            axis=0,
        )
        return jnp.sort(times_interp)

    def _lindblad_step(self, rho, h, cols):
        del_rho = -1j * self._commutator(h, rho)
        for col in cols:
            del_rho += jnp.matmul(jnp.matmul(col, rho), self._dagger(col))
            del_rho -= 0.5 * self._anti_commutator(jnp.matmul(self._dagger(col), col), rho)
        return del_rho

    @partial(jit, static_argnums=(0,))
    def _vern7_one_step(self, rho, h, col):
        k1 = self._lindblad_step(rho, h[0], col)
        k2 = self._lindblad_step(rho + (1 / 200) * k1, h[1], col)
        k3 = self._lindblad_step(rho + (-4361 / 4050) * k1 + (2401 / 2025) * k2, h[2], col)
        k4 = self._lindblad_step(
            rho + (49 / 1200) * k1 + (49 / 400) * k3,
            h[3],
            col,
        )
        k5 = self._lindblad_step(
            rho + (2454451729 / 3841600000) * k1 + (-9433712007 / 3841600000) * k3 + (4364554539 / 1920800000) * k4,
            h[4],
            col,
        )
        k6 = self._lindblad_step(
            rho
            + (-6187101755456742839167388910402379177523537620 / 2324599620333464857202963610201679332423082271) * k1
            + (27569888999279458303270493567994248533230000 / 2551701010245296220859455115479340650299761) * k3
            + (-37368161901278864592027018689858091583238040000 / 4473131870960004275166624817435284159975481033) * k4
            + (1392547243220807196190880383038194667840000000 / 1697219131380493083996999253929006193143549863) * k5,
            h[5],
            col,
        )
        k7 = self._lindblad_step(
            rho
            + (11272026205260557297236918526339 / 1857697188743815510261537500000) * k1
            + (-48265918242888069 / 1953194276993750) * k3
            + (26726983360888651136155661781228 / 1308381343805114800955157615625) * k4
            + (-2090453318815827627666994432 / 1096684189897834170412307919) * k5
            + (1148577938985388929671582486744843844943428041509 / 1141532118233823914568777901158338927629837500000)
            * k6,
            h[6],
            col,
        )
        k10 = self._lindblad_step(
            rho
            + (-511858190895337044664743508805671 / 11367030248263048398341724647960) * k1
            + (2822037469238841750 / 15064746656776439) * k3
            + (-23523744880286194122061074624512868000 / 152723005449262599342117017051789699) * k4
            + (10685036369693854448650967542704000000 / 575558095977344459903303055137999707) * k5
            + (
                -6259648732772142303029374363607629515525848829303541906422993
                / 876479353814142962817551241844706205620792843316435566420120
            )
            * k6
            + (17380896627486168667542032602031250 / 13279937889697320236613879977356033) * k7,
            h[8],
            col,
        )
        rho_new = (
            rho
            + (117807213929927 / 2640907728177740) * k1
            + (4758744518816629500000 / 17812069906509312711137) * k4
            + (1730775233574080000000000 / 7863520414322158392809673) * k5
            + (
                2682653613028767167314032381891560552585218935572349997
                / 12258338284789875762081637252125169126464880985167722660
            )
            * k6
            + (40977117022675781250 / 178949401077111131341) * k7
            + (2152106665253777 / 106040260335225546) * k10
        )
        return rho_new

    @partial(jit, static_argnums=(0,))
    def _propagate_in_time(self, rhos_t, eom, col, steps_arr):
        """
        Propagate from `time[ti] to time[ti+1]`.
        JIT compiled and uses `jax.lax.scan` to avoid compilation overhead.
        """

        def propagate_body(rhos_t, index):
            rhos_t = self._vern7_one_step(
                rhos_t,
                dynamic_slice_in_dim(eom, start_index=9 * index, slice_size=9, axis=0),
                col,
            )
            return rhos_t, rhos_t

        rhos_t, _ = scan(propagate_body, rhos_t, steps_arr)
        return rhos_t

    def propagate(self, time: Array):
        """Return the solution of the equation of motion for open system using vern7 ODE solver.

        Loop over all desired times in time at set resolution.

        Parameters
        ----------
        time : numpy.ndarray
            Any one-dimensional vector of timestamps.

        Returns
        -------
        jax.Array
            Returns the solution of the equations of motion.

        Raises
        ------
        cthree.Exceptions.ConfigurationException
            If the initial state is not set.

        """
        if self._initial_state is None:
            raise ConfigurationException("Initial state is not set")

        if len(time) < 2:
            raise ValueError("Propagation needs at least two time steps")

        init_state = jnp.array(self._initial_state, dtype=jnp.complex128)
        eom_func = self._model.get_matrix

        # Verify if `OpenSystem._ode_propagation` is set to `True`.
        # ode_propgation returns hamiltonian and collapse operators separately.
        eom_parts = eom_func(jnp.array([0]))
        if len(eom_parts) != 2:
            raise ConfigurationException(
                "Please set `OpenSystem.__ode_propagation` to `True` for this propagation method."
            )

        # Checking if initial state is a density matrix
        # Checking shapes at index 1 as index 0 can also be the "batch dimension"
        dim_generator = eom_parts[0].shape[1]
        if init_state.shape[1] != dim_generator:
            raise ConfigurationException(
                "Size mismatch between initial state and Hamiltonain. Initial state has to be a density matrix."
            )

        rhos = [init_state]

        for ti in range(1, len(time)):
            rhos_t = rhos[ti - 1]
            times, dt = self._construct_times(time, ti)
            times_interp = self._interpolate_time(times, dt)
            eom, cols = eom_func(times_interp + dt / 2)
            rhos_t = self._propagate_in_time(
                rhos_t,
                eom * dt,
                cols * jnp.sqrt(dt),
                jnp.arange(0, len(times), 1),
            )
            rhos.append(rhos_t)

        return jnp.array(rhos)
