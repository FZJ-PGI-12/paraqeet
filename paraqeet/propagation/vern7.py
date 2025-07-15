"""Class definition of the fixed time-step 7th-order Verner ODE solver. Adapted from Julia DiffEq Vern7."""

from functools import partial

import numpy as np
import jax.numpy as jnp

from paraqeet.exceptions import ConfigurationException
from paraqeet.quantity import Quantity, Array
from paraqeet.model.equation_of_motion import EquationOfMotion
from paraqeet.propagation.state_propagation import StatePropagation

import jax
from jax import jit
from jax.lax import scan, dynamic_slice_in_dim

jax.config.update("jax_enable_x64", True)


class Vern7(StatePropagation):
    """
    Propagate state by solving the Schrödinger equation / Lindblad master equation by using ODE solver.

    Implements Vern7 ODE Solver algorithm non adaptive (fixed time-step) version.
    """

    _res: float
    _initial_state: Array | None = None

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

        if self.is_open:
            self.step_function = self._lindblad_step
        else:
            self.step_function = self._schrodinger_step

    def set_initial_state(self, state):
        """Set initial state."""
        # For open system check if initial state is a density matrixs.
        if self.is_open:
            if state.shape[-1] != state.shape[-2]:
                raise ConfigurationException(
                    f"Obtained a state vector of shape {state.shape} as initial state. "
                    + "For open system propagation expected a density matrix as the initial state."
                )
        self._initial_state = jnp.array(state, dtype=jnp.complex128)

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
    def _commutator(A: Array, B: Array):
        return jnp.matmul(A, B) - jnp.matmul(B, A)

    @staticmethod
    def _anti_commutator(A: Array, B: Array):
        return jnp.matmul(A, B) + jnp.matmul(B, A)

    @staticmethod
    def _dagger(op: Array):
        return op.conj().T

    def _construct_times(self, time, ti):
        """Construct one-dimensional vector of time."""
        t0 = time[ti - 1]
        t1 = time[ti]
        steps = int(np.ceil((t1 - t0) * self._res))
        times = jnp.linspace(t0, t1, steps, endpoint=False)
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

    def _lindblad_step(self, state: Array, h: Array, cols: list[Array]):
        del_rho = self._commutator(h, state)
        for col in cols:
            del_rho += jnp.matmul(jnp.matmul(col, state), self._dagger(col))
            del_rho -= 0.5 * self._anti_commutator(jnp.matmul(self._dagger(col), col), state)
        return del_rho

    def _schrodinger_step(self, state: Array, h: Array, cols: list[Array]):
        return jnp.matmul(h, state)

    @partial(jit, static_argnums=(0,))
    def _vern7_one_step(self, state, h, col):
        k1 = self.step_function(state, h[0], col)
        k2 = self.step_function(state + (1 / 200) * k1, h[1], col)
        k3 = self.step_function(state + (-4361 / 4050) * k1 + (2401 / 2025) * k2, h[2], col)
        k4 = self.step_function(
            state + (49 / 1200) * k1 + (49 / 400) * k3,
            h[3],
            col,
        )
        k5 = self.step_function(
            state + (2454451729 / 3841600000) * k1 + (-9433712007 / 3841600000) * k3 + (4364554539 / 1920800000) * k4,
            h[4],
            col,
        )
        k6 = self.step_function(
            state
            + (-6187101755456742839167388910402379177523537620 / 2324599620333464857202963610201679332423082271) * k1
            + (27569888999279458303270493567994248533230000 / 2551701010245296220859455115479340650299761) * k3
            + (-37368161901278864592027018689858091583238040000 / 4473131870960004275166624817435284159975481033) * k4
            + (1392547243220807196190880383038194667840000000 / 1697219131380493083996999253929006193143549863) * k5,
            h[5],
            col,
        )
        k7 = self.step_function(
            state
            + (11272026205260557297236918526339 / 1857697188743815510261537500000) * k1
            + (-48265918242888069 / 1953194276993750) * k3
            + (26726983360888651136155661781228 / 1308381343805114800955157615625) * k4
            + (-2090453318815827627666994432 / 1096684189897834170412307919) * k5
            + (1148577938985388929671582486744843844943428041509 / 1141532118233823914568777901158338927629837500000)
            * k6,
            h[6],
            col,
        )
        k10 = self.step_function(
            state
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
        state_new = (
            state
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
        return state_new

    @partial(jit, static_argnums=(0,))
    def _propagate_in_time(self, state_t, eom, col, steps_arr):
        """
        Propagate from `time[ti] to time[ti+1]`.
        JIT compiled and uses `jax.lax.scan` to avoid compilation overhead.
        """

        def propagate_body(state_t, index):
            state_t = self._vern7_one_step(
                state_t,
                dynamic_slice_in_dim(eom, start_index=9 * index, slice_size=9, axis=0),
                col,
            )
            return state_t, state_t

        state_t, _ = scan(propagate_body, state_t, steps_arr)
        return state_t

    def propagate(self, time: Array) -> Array:
        """Return the solution of the equation of motion for open/closed system using vern7 ODE solver.

        Loop over all desired times in time at set resolution.

        Parameters
        ----------
        time : Array
            Any one-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the solution of the equations of motion.

        Raises
        ------
        ConfigurationException
            If the initial state is not set.

        """
        if self._initial_state is None:
            raise ConfigurationException("Initial state is not set")

        if len(time) < 2:
            raise ValueError("Propagation needs at least two time steps")

        init_state = jnp.array(self._initial_state, dtype=jnp.complex128)
        if self._model is None:
            raise ConfigurationException("No equation of motion is configured.")
        eom_func = self._model.get_matrix

        # Verify if `model.ode_propagation` is set to `True`.
        # ode_propgation returns hamiltonian and collapse operators separately.
        eom_parts = eom_func(jnp.array([0]))
        if len(eom_parts) != 2:
            raise ConfigurationException("Please set `model.ode_propagation` to `True` for this propagation method.")

        states = [init_state]

        for ti in range(1, len(time)):
            state_t = states[ti - 1]
            times, dt = self._construct_times(time, ti)
            times_interp = self._interpolate_time(times, dt)
            eom, cols = eom_func(times_interp + dt / 2)
            state_t = self._propagate_in_time(
                state_t,
                eom * dt,
                jnp.array(cols) * jnp.sqrt(dt),
                jnp.arange(0, len(times), 1),
            )
            states.append(state_t)

        return jnp.array(states)
