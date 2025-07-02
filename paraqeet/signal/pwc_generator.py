from functools import partial

import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import erf

from paraqeet.quantity import Quantity, Array
from paraqeet.signal.waveform import Waveform
from paraqeet.signal.generator import Generator


class PWCGenerator(Generator):
    """Convert a complex envelope to PWC pulse.

    This sets the pulse parameters to the `tlist` points.
    The gradient of the pulse wrt the PWC bins is 1 at that time point and zero
    everywhere else.

    This Generator doesn't add the LO signal to the envelope pulse.
    Driving with a PWC pulse (without the LO) should be done in the rotating
    frame of drive.

    __envs: list[Waveform]
        List of Envelopes
    __tlist: Array
        Time grid discritization points

    Parameters
    ----------
    envelopes : List[Waveform]
        List of input devices.

    """

    __envs: list[Waveform]
    __tlist: Array
    __inphase: Quantity
    __quadrature: Quantity
    _optimisable_parameters: list[Quantity] = []
    __multiply_flat_top: bool = False

    def __init__(
        self,
        envelopes: list[Waveform] | None,
        tlist: Array,
    ):
        self.__envs = envelopes or []
        self.__tlist = tlist

        # Choose the center point as time grid
        dt = tlist[1] - tlist[0]
        self.__tlist = tlist[:-1] + dt / 2

        self.__setup_inphase_and_quadrature()
        self.__t_final = self.__tlist[-1]

    @partial(jit, static_argnums=(0,))
    def __compute_envelope(self, t):
        t_final = self.__t_final
        ramp_time = t_final / 25
        rampUp = 1 + erf((t - 2 * t_final / 20) / ramp_time)
        rampDown = 1 + erf((-t + 18 * t_final / 20) / ramp_time)
        return rampUp * rampDown / 4

    @property
    def tlist(self) -> Array:
        """Get time grid discritization for generating PWC pulse.

        Returns
        -------
        Array
            Array of time points at which envelope is discritized.
        """
        return self.__tlist

    @tlist.setter
    def tlist(self, tlist: Array) -> None:
        """Set time grid discritization for generating PWC pulse.

        Parameters
        ----------
        tlist : Array
            Array of time points at which envelope is discritized.
        """
        self.__tlist = tlist
        self.__setup_inphase_and_quadrature()

    @property
    def multiply_flat_top(self) -> bool:
        """Flag to multiply the pulse with a FlatTop.

        This can be used to make the start and end values zeros and force the
        PWC pulse to change smoothly.

        Returns
        -------
        multiply_flat_top : bool
            Flag value for multiply_flat_top.
        """
        return self.__multiply_flat_top

    @multiply_flat_top.setter
    def multiply_flat_top(self, multiply_flat_top: bool) -> None:
        """Set flag to multiply the pulse with a FlatTop.

        This can be used to make the start and end values zeros and force the
        PWC pulse to change smoothly.

        Parameters
        ----------
        multiply_flat_top : bool
            Flag value for multiply_flat_top.
        """
        self.__multiply_flat_top = multiply_flat_top
        self.__setup_inphase_and_quadrature()

    def __compute_shape(self) -> Array:
        env = jnp.zeros_like(self.__tlist)
        for dev in self.__envs:
            env += dev.compute_output(self.__tlist)
        return env

    def __setup_inphase_and_quadrature(self) -> None:
        """Generate Inphase and Quadrature Quantities using tlist."""
        env = self.__compute_shape()

        max_abs = jnp.max(jnp.abs(env))
        bound = 2 * max_abs * jnp.ones_like(self.__tlist)

        self.__inphase = Quantity(
            jnp.real(env),
            min_value=-bound,
            max_value=bound,
            unit="Hz",
            name="Inphase",
        )
        self.__quadrature = Quantity(
            jnp.imag(env),
            min_value=-bound,
            max_value=bound,
            unit="Hz",
            name="Quadrature",
        )

    def _get_partial_derivatives(self) -> Array:
        env_grads = []
        for dev in self.__envs:
            grad = dev.compute_gradient(self.__tlist)
            dev_grad = jnp.concat([jnp.real(grad), jnp.imag(grad)])
            env_grads.append(dev_grad)
        return jnp.hstack(env_grads)

    def _update_inphase_and_quadrature(self) -> None:
        env = self.__compute_shape()
        self.__inphase.set_value(jnp.real(env))
        self.__quadrature.set_value(jnp.imag(env))

    def get_parameters(self) -> list[Quantity]:
        """Return a list of parameters.

        Return the inphase and quadrature as parameters.

        Returns
        -------
        List[Quantity]
            All Parameters describing the signal.
        """
        return [self.__inphase, self.__quadrature]

    def set_optimisable_parameters(self, params: list[Quantity]) -> None:
        """Set specified parameters to be optimised.

        Optimisable paramters can be inphase and quadrature.

        Parameters
        ----------
        params : list[Quantity]
        """
        super().set_optimisable_parameters(params)

    @partial(jit, static_argnums=(0,))
    def __pwc_signal(
        self,
        inphase: Array,
        quadrature: Array,
        tlist: Array,
        t: Array,
    ) -> Array:
        """Generate a signal for a single time point 't'.

        The PWC signal is generated by finiding the closest time point
        and returning the correspoinding amplitude value.

        Parameters
        ----------
        inphase: Array
            1-D vector of step values of real part of the PWC signal.
        quadrature: Array
            1-D vector of step values of complex part of the PWC signal.
        tlist: Array
            Time bins of the PWC pulse.
        t : float
            One time point.

        Returns
        -------
        Array
            Returns the PWC signal value at t.

        """
        index = jnp.argmin(jnp.abs(tlist - t))
        return inphase[index] + 1j * quadrature[index]

    def generate_signal(self, times: Array) -> Array:
        """Generate the PWC signal for time(s) 't'.

        Parameters
        ----------
        t : Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the signal vector.

        """
        tlist = self.__tlist
        t_arr = jnp.array(times, ndmin=1)
        inphase = self.__inphase.get_value()
        quadrature = self.__quadrature.get_value()
        if self.__multiply_flat_top:
            env = self.__compute_envelope(tlist)
            inphase *= env
            quadrature *= env
        shape = jnp.squeeze(vmap(self.__pwc_signal, in_axes=(None, None, None, 0))(inphase, quadrature, tlist, t_arr))
        return shape

    def generate_signal_gradient(self, times: Array) -> Array:
        """Return signal gradient wrt inphase and quadrature.

        This returns a list of ones as the gradient of the envelope wrt a step
        is 1 for that time bin and 0 everywhere else.

        Parameters
        ----------
        t : Array
            Array of time steps.

        Returns
        -------
        Array
            PWC signal gradients.
        """
        t_arr = jnp.array(times, ndmin=1)

        grads = []
        tlist = self.__tlist

        if self.__multiply_flat_top:
            smoothing = self.__compute_envelope(tlist)
            index = jnp.argmin(jnp.abs(tlist - t_arr))
            env = smoothing[index]
        else:
            env = jnp.ones_like(t_arr)

        if self._is_optimised(self.__inphase):
            grads.append(env)
        if self._is_optimised(self.__quadrature):
            grads.append(1j * env)

        if len(grads) > 0:
            grads_stack = jnp.stack(grads)
        else:
            grads_stack = jnp.empty((t_arr.shape[0], 0))
        return grads_stack

    def generate_signal_gradient_one_time(self, time: Array) -> Array:
        """Return signal gradient wrt inphase and quadrature.

        This returns a list of ones as the gradient of the envelope wrt a step
        is 1 for that time bin and 0 everywhere else.

        Parameters
        ----------
        t : float
            One time step.

        Returns
        -------
        Array
            PWC signal gradients.
        """
        grads = []
        tlist = self.__tlist

        if self.__multiply_flat_top:
            smoothing = self.__compute_envelope(tlist)
            index = jnp.argmin(jnp.abs(tlist - time))
            env = smoothing[index]
        else:
            env = 1

        if self._is_optimised(self.__inphase):
            grads.append(env)
        if self._is_optimised(self.__quadrature):
            grads.append(1j * env)

        return jnp.stack(grads, axis=0) if len(grads) > 0 else jnp.empty((0,))
