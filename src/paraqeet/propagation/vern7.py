"""Class definition of the fixed time-step 7th-order Verner ODE solver. Adapted from Julia DiffEq Vern7."""

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import jit
from jax.lax import dynamic_slice_in_dim, scan

from paraqeet.propagation.propagation import StatePropagation
from paraqeet.propagation.utils import construct_times
from paraqeet.quantity import Array

jax.config.update("jax_enable_x64", True)


class Vern7(StatePropagation):
    """
    Propagate state by solving the Schrödinger equation / Lindblad master equation by using ODE solver.

    Implements Vern7 ODE Solver algorithm :cite:p:`verner2010numerically` non adaptive (fixed time-step) version.
    """

    _step_function: Callable
    _jump_operators: Array

    def __init__(
        self,
        eom_func: Callable[[Array], Array],
        resolution: float,
        initial_state: Array,
        step_function: Callable,
        jump_operators: list[Array] | None = None,
    ) -> None:
        """
        Args:
            eom_func: Equation of motion (EOM) as a function of time.
            resolution: Resolution at which to sample the EOM.
            initial_state: Initial state.
            step_function: Step function used to that implements the right hand side of the EOM.
            jump_operators: A list of jump operators (each multiplied by the sqrt of the corresponding decay rate).
                Defaults to None for closed system.
        """
        super().__init__(eom_func, resolution, initial_state)
        self._step_function = step_function
        self.jump_operators = jump_operators

    @property
    def step_function(self) -> Callable:
        """Return the step function for solving the EOM."""
        return self._step_function

    @step_function.setter
    def step_function(self, step_func: Callable) -> None:
        """Set the step function for solving the EOM."""
        self._step_function = step_func

    @property
    def jump_operators(self) -> Array:
        """Return the jump operators used for solving the EOM."""
        return self._jump_operators

    @jump_operators.setter
    def jump_operators(self, jump_ops: list[Array] | None) -> None:
        """Set the jump operators added to the EOM."""
        if jump_ops is not None:
            self._jump_operators = jnp.array(jump_ops)
        else:
            self._jump_operators = jnp.empty((0,) + self._eom_func(jnp.array([0.0])).shape)

    @staticmethod
    def _interpolate_time(times: Array, dt: Array | float) -> Array:
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

    @partial(jit, static_argnums=(0,))
    def _vern7_one_step(self, state: Array, h: Array, col: Array) -> Array:
        k1 = self._step_function(state, h[0], col)
        k2 = self._step_function(state + (1 / 200) * k1, h[1], col)
        k3 = self._step_function(state + (-4361 / 4050) * k1 + (2401 / 2025) * k2, h[2], col)
        k4 = self._step_function(
            state + (49 / 1200) * k1 + (49 / 400) * k3,
            h[3],
            col,
        )
        k5 = self._step_function(
            state + (2454451729 / 3841600000) * k1 + (-9433712007 / 3841600000) * k3 + (4364554539 / 1920800000) * k4,
            h[4],
            col,
        )
        k6 = self._step_function(
            state
            + (-6187101755456742839167388910402379177523537620 / 2324599620333464857202963610201679332423082271) * k1
            + (27569888999279458303270493567994248533230000 / 2551701010245296220859455115479340650299761) * k3
            + (-37368161901278864592027018689858091583238040000 / 4473131870960004275166624817435284159975481033) * k4
            + (1392547243220807196190880383038194667840000000 / 1697219131380493083996999253929006193143549863) * k5,
            h[5],
            col,
        )
        k7 = self._step_function(
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
        k10 = self._step_function(
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
        state_new: Array = (
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
    def _propagate_in_time(self, state_t: Array, eom: Array, col: Array, steps_arr: Array) -> Array:
        """
        Propagate from ``time[ti]`` to ``time[ti+1]``.
        JIT compiled and uses ``jax.lax.scan`` to avoid compilation overhead.
        """

        def propagate_body(state_t: Array, index: Any) -> tuple[Array, Array]:
            state_t = self._vern7_one_step(
                state_t,
                dynamic_slice_in_dim(eom, start_index=9 * index, slice_size=9, axis=0),
                col,
            )
            return state_t, state_t

        state_t, _ = scan(propagate_body, state_t, steps_arr)
        return state_t

    def get_value(self, times: Array) -> Array:
        """Return the solution of the equation of motion for open/closed system using vern7 ODE solver.

        Loop over all desired times in time at set resolution.

        Args:
            times: Array of times.

        Returns:
            The solution of the equations of motion.

        Raises:
            ValueError: If fewer than two time points are given.
        """
        if len(times) < 2:
            raise ValueError("Vern7.get_value needs at least two time points.")

        init_state = jnp.array(self._initial_state, dtype=jnp.complex128)

        states = [init_state]
        for ti in range(1, len(times)):
            state_t = states[ti - 1]
            step_times, dt = construct_times(times, ti, self._resolution)
            times_interp = Vern7._interpolate_time(step_times, dt)
            # TODO: Separate jump operators from EOM.
            eom = self._eom_func(times_interp + dt / 2)
            state_t = self._propagate_in_time(
                state_t,
                eom * dt,
                self._jump_operators * jnp.sqrt(dt),
                jnp.arange(0, len(step_times), 1),
            )
            states.append(state_t)

        return jnp.array(states)
