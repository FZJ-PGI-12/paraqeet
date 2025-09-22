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
    Driving with a PWC pulse (without the LO) is usually done in the rotating
    frame of drive.

    This Generator converts the input complex pulse to the 'in-phase' and
    'out-of-phase' components. This naming convention is used by following [Krantz2019].
    In the literature of signal processing these are also called 'in-phase' and 'quadrature'
    components (refer to https://en.wikipedia.org/wiki/In-phase_and_quadrature_components).

    [Krantz2019] Krantz et al., “A Quantum Engineer’s Guide to Superconducting Qubits.” Applied Physics Reviews 6(2019).

    __envs: list[Waveform]
        List of Envelopes
    __tlist: Array
        Time grid discritization points
    __max_amplitude: float
        Maximum amplitude of the drive
    __inphase: Quantity
        The in-phase component of the pulse
    __outofphase: Quantity
        The out-of-phase component of the pulse
    _optimisable_paramters: list[Quantity]
        List of own parameters that would be optimised by the optimiser.
    __multiply_flat_top: bool
        Flag to multiply flat-top-Gaussain pulse to the signal to ensure it
        starts and ends at zero.

    Parameters
    ----------
    envelopes : List[Waveform]
        List of input devices.

    """

    __envs: list[Waveform]
    __tlist: Array
    __max_amplitude: float
    __inphase: Quantity
    __outofphase: Quantity
    _optimisable_parameters: list[Quantity] = []
    __multiply_flat_top: bool = False

    def __init__(
        self,
        envelopes: list[Waveform] | None,
        tlist: Array,
        max_amplitude: float | None = None,
    ):
        self.__envs = envelopes or []
        self.__tlist = tlist

        if max_amplitude is not None and max_amplitude < 0.0:
            raise ValueError("The maximum drive amplitude must be positive.")
        elif max_amplitude is None:
            env = self.__compute_shape()
            self.__max_amplitude = 2 * float(jnp.max(jnp.abs(env)))
        else:
            self.__max_amplitude = max_amplitude

        # Choose the center point as time grid
        dt = tlist[1] - tlist[0]
        self.__tlist = tlist[:-1] + dt / 2

        self.__setup_inphase_and_outofphase()
        self.__t_final = self.__tlist[-1]

    @partial(jit, static_argnums=(0,))
    def __compute_envelope(self, t):
        t_final = self.__t_final
        ramp_time = t_final / 25
        ramp_up = 1 + erf((t - 2 * t_final / 20) / ramp_time)
        ramp_down = 1 + erf((-t + 18 * t_final / 20) / ramp_time)
        return ramp_up * ramp_down / 4

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
        self.__setup_inphase_and_outofphase()

    @property
    def max_amplitude(self) -> float:
        """Get the maximum drive amplitude.

        Returns
        -------
            The value of the maximum drive amplitude.
        """
        return self.__max_amplitude

    @max_amplitude.setter
    def max_amplitude(self, max_amplitude: float) -> None:
        """Set the maximum drive amplitude of the drive.

        Parameters
        ----------
        max_amplitude: float
            The value of the maximu drive amplitude.
        """
        self.__max_amplitude = max_amplitude
        self.__setup_inphase_and_outofphase()

    @property
    def envs(self) -> list[Waveform]:
        """Gets the list of envelopes.

        Returns
        -------
        list[Waveform]
            The list of waveforms associated with the generator.
        """
        return self.__envs

    @property
    def multiply_flat_top(self) -> bool:
        """Flag to multiply the pulse with a FlatTop.

        This can be used to make the start and end values zeros and force the
        PWC pulse to change smoothly.

        Returns
        -------
        multiply_flat_top: bool
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
        self.__setup_inphase_and_outofphase()

    def __compute_shape(self) -> Array:
        env = jnp.zeros_like(self.__tlist)
        for dev in self.__envs:
            env += dev.compute_output(self.__tlist)
        return env

    def __setup_inphase_and_outofphase(self) -> None:
        """Generate inphase and outphase Quantities using tlist."""
        env = self.__compute_shape()

        # max_abs = jnp.max(jnp.abs(env))
        max_component = self.__max_amplitude / jnp.sqrt(2)
        bound = max_component * jnp.ones_like(self.__tlist)

        self.__inphase = Quantity(
            jnp.real(env),
            min_value=-bound,
            max_value=bound,
            name="Inphase",
        )
        self.__outofphase = Quantity(
            jnp.imag(env),
            min_value=-bound,
            max_value=bound,
            name="out-of-phase",
        )

    def _get_partial_derivatives(self) -> Array:
        env_grads = []
        for dev in self.__envs:
            grad = dev.compute_gradient(self.__tlist)
            dev_grad = jnp.concat([jnp.real(grad), jnp.imag(grad)])
            env_grads.append(dev_grad)
        return jnp.hstack(env_grads)

    def _update_inphase_and_outofphase(self) -> None:
        env = self.__compute_shape()
        self.__inphase.set_value(jnp.real(env))
        self.__outofphase.set_value(jnp.imag(env))

    def get_parameters(self) -> list[Quantity]:
        """Return a list of parameters.

        Return the inphase and out-of-phase as parameters.

        Returns
        -------
        list[Quantity]
            All Parameters describing the signal.
        """
        return [self.__inphase, self.__outofphase]

    def set_optimisable_parameters(self, params: list[Quantity]) -> None:
        """Set specified parameters to be optimised.

        Optimisable paramters can be inphase and out-of-phase.

        Parameters
        ----------
        params : list[Quantity]
        """
        super().set_optimisable_parameters(params)

    @partial(jit, static_argnums=(0,))
    def __pwc_signal(
        self,
        inphase: Array,
        outofphase: Array,
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
        out-of-phase: Array
            1-D vector of step values of complex part of the PWC signal.
        tlist: Array
            Time bins of the PWC pulse.
        t: Array
            One time point.

        Returns
        -------
        Array
            Returns the PWC signal value at t.

        """
        index = jnp.argmin(jnp.abs(tlist - t))
        return inphase[index] + 1j * outofphase[index]

    def generate_signal(self, times: Array) -> Array:
        """Generate the PWC signal for time(s) 't'.

        Parameters
        ----------
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the signal vector.

        """
        tlist = self.__tlist
        t_arr = jnp.array(times, ndmin=1)
        inphase = self.__inphase.get_value()
        outofphase = self.__outofphase.get_value()
        if self.__multiply_flat_top:
            env = self.__compute_envelope(tlist)
            inphase *= env
            outofphase *= env
        shape = jnp.squeeze(vmap(self.__pwc_signal, in_axes=(None, None, None, 0))(inphase, outofphase, tlist, t_arr))
        return shape

    def generate_signal_gradient(self, times: Array) -> Array:
        """Return signal gradient wrt inphase and out-of-phase.

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
        if self._is_optimised(self.__outofphase):
            grads.append(1j * env)

        if len(grads) > 0:
            grads_stack = jnp.stack(grads)
        else:
            grads_stack = jnp.empty((t_arr.shape[0], 0))
        return grads_stack

    def generate_signal_gradient_one_time(self, time: Array) -> Array:
        """Return signal gradient wrt inphase and out-of-phase.

        This returns a list of ones as the gradient of the envelope wrt a step
        is 1 for that time bin and 0 everywhere else.

        Parameters
        ----------
        time: Array
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
        if self._is_optimised(self.__outofphase):
            grads.append(1j * env)

        return jnp.stack(grads, axis=0) if len(grads) > 0 else jnp.empty((0,))
