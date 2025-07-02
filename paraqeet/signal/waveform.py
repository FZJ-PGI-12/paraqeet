"""Class definition for the Device model."""

from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Any

import jax.numpy as jnp
from paraqeet.quantity import Array
from paraqeet.exceptions import ConfigurationException
from jax import grad, jit, vmap
from jax.scipy.special import erf

from paraqeet.optimisable import Optimisable
from paraqeet.quantity import Quantity

import jax

jax.config.update("jax_enable_x64", True)


class Waveform(Optimisable):
    """Classical electronics."""

    _gradient_function: Callable | None = None
    _grad_arg_nums: tuple[int, ...] = ()

    def _compute_gradient_function(
        self, signalFunction: Callable, argnums: tuple[int, ...], vmap_axes: tuple[int | None, ...]
    ):
        """Return a compute gradient function from the signal function.

        Parameters
        ----------
        signalFunction : Callable
            A function that generated signals.
        argnums : Tuple[int, ...]
            A tuple of ints containing a variable number of argument numbers.
        vmap_axes : Tuple[int, ...]
            A tuple of ints.

        """
        grads = grad(signalFunction, argnums=argnums)
        partial_grads = vmap(grads, vmap_axes)
        self._gradient_function = jit(partial_grads)

    def set_optimisable_parameters(self, params: list[Quantity]) -> None:
        """Set optimisable parameters for optimisation.

        Parameters
        ----------
        params : List[Quantity]
            Input list of parameters to be set.

        """
        super().set_optimisable_parameters(params)

        self._grad_arg_nums = ()
        for i, param in enumerate(self.get_parameters()):
            if self._is_optimised(param):
                self._grad_arg_nums += (i,)

        # Recompute gradient function
        params = self.get_parameters()
        num_params = len(params)

        # vmap over time axis only, set everything else to None
        vmap_axes = (None,) * num_params
        vmap_axes += (0,)  # type: ignore

        if len(self._grad_arg_nums) > 0:
            self._compute_gradient_function(
                self._evaluate,
                argnums=self._grad_arg_nums,
                vmap_axes=vmap_axes,
            )
        else:
            self._gradient_function = None

    @abstractmethod
    def _evaluate(self, *args, **kwargs) -> Array:
        """Evaluate the output of the system.

        Abstract method.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    @abstractmethod
    def compute_output(self, t: Array) -> Array:
        """Compute the output.

        Parameters
        ----------
        t : Array or float
            One-dimensional vector of timestamps or a single value.

        Returns
        -------
        Array
            Output of the computation.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    def compute_gradient(self, t: Array) -> Array:
        """Compute the gradient of the `_evaluate` method.

        Uses Automatic differentiation.
        The `_evaluate` method should be a `pure` function (should take the
        optimisable parameters as function arguments and doesn't depend on
        global variables).
        Refer to https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
        for functionally `pure` functions.
        To implement analytical gradients / other methods for gradient
        computation overwrite this method in the inherited class.

        Parameters
        ----------
        t : Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the gradient array of the `_evaluate` method.

        """
        params = self.get_parameters()
        param_values = [param.get_value() for param in params]
        t_arr = jnp.array(t, ndmin=1)
        grads = jnp.empty((t_arr.shape[0], 0))
        if self._gradient_function is not None:
            grads = jnp.stack(self._gradient_function(*param_values, t_arr), axis=1)
            grads = jnp.squeeze(grads, -1)
        return grads

    def compute_time_gradient(self, t: Array) -> Array:
        """Compute a signal envelopes time derivative.

        Parameters
        ----------
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns a vector signals time derivative.

        """
        t_arr = jnp.array(t, ndmin=1)
        envTimeGradFun = grad(self.compute_output, argnums=0)
        envTimeGrad = vmap(envTimeGradFun, in_axes=(0,))(t_arr)
        return jnp.squeeze(envTimeGrad)


class LocalOscillator(Waveform):
    """A local oscillators carrier signal.

    __lo_freq : Quantity
        The frequency of the carrier signal.

    """

    __lo_freq: Quantity

    def __init__(self, frequency: Quantity | None = None) -> None:
        self.__lo_freq = frequency or Quantity(
            value=jnp.array(4.8e9 * 2 * jnp.pi),
            min_value=jnp.array(0.8 * 4.8e9 * 2 * jnp.pi),
            max_value=jnp.array(1.2 * 4.8e9 * 2 * jnp.pi),
            unit="Hz",
            name="lo_freq",
            two_pi=True,
        )

    def get_parameters(self) -> list[Quantity]:
        """Return device parameters.

        Returns
        -------
        list[Quantity]
            Returns the carrier signal frequency.
        """
        return [self.__lo_freq]

    @property
    def frequency(self) -> Quantity:
        """Get The frequency of the constant oscillating tone.

        Returns
        -------
        Quantity
            The frequency of the tone.

        """
        return self.__lo_freq

    @frequency.setter
    def frequency(self, frequency: Quantity) -> None:
        """Set The frequency of the constant oscillating tone.

        Parameters
        ----------
        freq : Quantity
            The frequency of the constant oscillating tone.

        """
        self.__lo_freq = frequency

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate(self, freq: Array, t: Array) -> Array:  # type: ignore
        """Calculate the unscaled carrier signal.

        Parameters
        ----------
        freq : Array
            The frequency of the carrier signal
        t : Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            The unscaled the carrier signal.
        """
        return jnp.exp(1j * freq * t)

    def compute_output(self, t: Array | float) -> Array:
        """Evaluate a carrier signal from an input time vector.

        Parameters
        ----------
        t : Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns a vector carrier signal.
        """
        return self._evaluate(self.__lo_freq.get_value(), t)  # type: ignore

    def compute_gradient(self, t: Array) -> Array:
        """Return the gradient wrt to frequency of carrier signal.

        Parameters
        ----------
        t : Array
            Array of time points to evaluate gradients at.

        Returns
        -------
        Array
            Gradient of tone wrt to frequency.
        """
        freq = self.__lo_freq.get_value()
        t_arr = jnp.array(t, ndmin=1)

        grads = jnp.empty((t.shape[0], 0))
        if self._is_optimised(self.__lo_freq):
            grads = jnp.reshape(1j * t_arr * self._evaluate(freq, t_arr), (-1, 1))

        return grads

    def compute_time_gradient(self, t: Array) -> Array:
        """Compute a signals time derivative.

        Parameters
        ----------
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array or JitWrapped
            Returns a vector signals time derivative.

        """
        freq = self.__lo_freq.get_value()
        # returns JitWrapped
        return 1j * t * self._evaluate(freq, t)  # type: ignore


class DRAGMixer(Waveform):
    """A DRAG mixed waveform signal.

    The DRAG component is calculated for a set of envelopes and added in
    orthogonal direction in the x-y plane.

    __envs: list[Envelope]
        The list of shape defining signal envelops.
    __deltas: list[Quantity]
        The delta parameter by which to shift the frequency of the DRAG
        component.

    """

    __multiply_flat_top: bool = False

    def __init__(
        self,
        envelopes: Waveform | list[Waveform],
        deltas: list[Quantity] | None = None,
        t_final: Quantity | None = None,
    ) -> None:
        self.__envs = envelopes if isinstance(envelopes, list) else [envelopes]
        self.__add_deltas(self.__envs, deltas)
        self.__t_final = t_final

    @property
    def multiply_flat_top(self) -> bool:
        """Flag to multiply the pulse with a FlatTop.

        This can be used to make the start and end values zeros and force the
        pulse to change smoothly.

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
        pulse to change smoothly.

        Parameters
        ----------
        multiply_flat_top : bool
            Flag value for multiply_flat_top.
        """
        self.__multiply_flat_top = multiply_flat_top
        if self.__t_final is None:
            raise ConfigurationException("`t_final` is set to None. Specify pulse length to use `multiply_flat_top`")

    def get_parameters(self) -> list[Quantity]:
        """Return a list of parameters.

        Collects and returns a list of parameters from the tone, generator
        and the carrier signal.

        Returns
        -------
        List[Quantity]
            All Parameters describing the signal.
        """
        params = list()
        for tone in self.__envs:
            params += tone.get_parameters()
            params += [self.__get_tone_delta(tone)]
        return params

    @staticmethod
    def __add_deltas(envelopeTones: list[Waveform], deltas: list[Quantity] | None) -> None:
        """Add a DRAG delta parameter Quantity to each envelope Tone.

        Parameters
        ----------
        envelopeTones : List[Waveform]
            The list of tones defining the total envelope.
        deltas : List[Quantity]
            A List of Quantities representing the delta parameters to add to
            each envelope Tone.

        Returns
        -------
        List[Waveform]
            The list of envelope Tones with the added delta parameters.
        """
        for ii, env_tone in enumerate(envelopeTones):
            env_tone.__setattr__(
                "_" + env_tone.__class__.__name__ + "__delta",
                Quantity(
                    deltas[ii].get_value() if deltas else jnp.array(-200e6 * 2 * jnp.pi),
                    min_value=jnp.array(-3 * 200e6 * 2 * jnp.pi),
                    max_value=jnp.array(-0.1 * 200e6 * 2 * jnp.pi),
                    unit="Hz",
                    name="Delta",
                ),
            )

    @staticmethod
    def __get_tone_delta(tone: Waveform) -> Any:
        """Return a list of deltas for each tone.

        Returns
        -------
        Quantity
            List of delta values for each tone.
        """
        return tone.__getattribute__("_" + tone.__class__.__name__ + "__delta")

    @partial(jit, static_argnums=(0,))
    def __compute_flat_top_envelope(self, t):
        t_final = self.__t_final.get_value()
        ramp_time = t_final / 25
        rampUp = 1 + erf((t - 2 * t_final / 20) / ramp_time)
        rampDown = 1 + erf((-t + 18 * t_final / 20) / ramp_time)
        return rampUp * rampDown / 4

    def _evaluate(self, t, *deltas) -> Array:
        """Compute the DRAG Envelope using deltas.

        Explicit function depending on deltas to compute gradients using AD.

        Parameters
        ----------
        t : Array
            One-dimensional vector of timestamps.
        deltas: List[float]
            Variable number of inputs for delta parameters for each tone.

        Returns
        -------
        Array
            Returns a vector signal of the DRAG envelope.
        """
        total_env = jnp.zeros_like(t, dtype=jnp.complex128)
        for delta, tone in zip(deltas, self.__envs):
            env = tone.compute_output(t)
            env_grad = tone.compute_time_gradient(t)
            total_env += env - 1.0j / delta * env_grad
        if self.multiply_flat_top:
            flattop_env = self.__compute_flat_top_envelope(t)
            total_env *= flattop_env
        return jnp.squeeze(total_env)

    def compute_output(self, t: Array | float) -> Array:
        """Evaluate a carrier signal from an input time vector.

        Parameters
        ----------
        t : Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns a vector carrier signal.
        """
        deltas = [self.__get_tone_delta(tone).get_value() for tone in self.__envs]
        return self._evaluate(t, *deltas)

    def set_optimisable_parameters(self, params: list[Quantity]) -> None:
        """Set specified parameters to be optimised.

        Also add the indices to `__grad_arg_nums` to compute the gradients.

        Parameters
        ----------
        params : list[Quantity]
        """
        super().set_optimisable_parameters(params)

        for tone in self.__envs:
            tone.set_optimisable_parameters(params)

    def compute_gradient(self, t: Array) -> Array:
        """Generate gradient of the signal for an array of time.

        Collect and return the parameter gradients from the Tone and the carrier
        Tone. Compute the gradient of the generator parameters by AD.
        The order of the gradients should match the order of paramters in
        `self.getParameter()` method

        Parameters
        ----------
        t : Array
            An array of time points.

        Returns
        -------
        Array
            Array of gradients wrt each parameter for each time point.
        """
        deltas = [self.__get_tone_delta(tone) for tone in self.__envs]
        delta_values = [delta.get_value() for delta in deltas]

        gradients = jnp.zeros(shape=(t.shape[0], 0))

        if self.multiply_flat_top:
            smoothing = self.__compute_flat_top_envelope(t)
        else:
            smoothing = jnp.ones_like(t)

        # Collect gradients wrt envelope parameters
        for tone in self.__envs:
            grads = tone.compute_gradient(t)
            gradients = jnp.append(gradients, grads * jnp.expand_dims(smoothing, axis=1), axis=1)

        # Collect gradients wrt deltas
        for i, tone in enumerate(self.__envs):
            if self._is_optimised(deltas[i]):
                grad = 1j / (delta_values[i] ** 2) * tone.compute_time_gradient(t)
                grad = jnp.expand_dims(grad * smoothing, axis=1)
                gradients = jnp.append(gradients, grad, axis=1)

        return jnp.array(gradients)
