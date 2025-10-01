"""Class definition for the Evelopes."""

from abc import abstractmethod
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from paraqeet.quantity import Array
from jax import jit
from jax.scipy.special import erf

from paraqeet.quantity import Quantity
from paraqeet.signal.waveform import Waveform

import time

jax.config.update("jax_enable_x64", True)


class Envelope(Waveform):
    """Classical Signal Envelope class.

    _amplitude: Quantity
        The amplitude of the envelope.
    _t_final: Quantity
        The length in time of the envelope.
    _gradient_function: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _grad_arg_nums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.

    """

    _amplitude: Quantity
    _t_final: Quantity

    def __init__(
        self,
        amplitude: Quantity | None = None,
        t_final: Quantity | None = None,
    ):
        self._amplitude = amplitude or Quantity(
            1.55e8,
            min_value=jnp.array(0.0),
            max_value=jnp.array(1e9),
            unit="Hz",
            name="Amplitude",
            two_pi=True,
        )

        self._t_final = t_final or Quantity(
            32e-9,
            min_value=jnp.array(0),
            max_value=jnp.array(100e-9),
            unit="s",
            name="t_final",
        )

        self._gradient_function: Callable | None = None
        self._grad_arg_nums: tuple[int, ...] = ()

    def get_parameters(self):
        """Get a list of parameters of the envelope.

        Returns
        -------
        List[Quantity]
            List of parameters of the envelope.

        """
        return [self._amplitude, self._t_final]

    @property
    def amplitude(self) -> Quantity:
        """Get the amplitude of the system.

        Returns
        -------
        Quantity
            Amplitude of the system.

        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, amplitude: Quantity) -> None:
        """Set the amplitude of the system.

        Parameters
        ----------
        Quantity
            Amplitude value of the system to be set.

        """
        self._amplitude = amplitude

    @property
    def t_final(self) -> Quantity:
        """Get the length of the tone.

        Returns
        -------
        Quantity
            Length in time of the tone.

        """
        return self._t_final

    @t_final.setter
    def t_final(self, t_final: Quantity) -> None:
        """Set the length of the tone.

        Parameters
        ----------
        Quantity
            Length in time of the tone to be set.

        """
        self._t_final = t_final

    @abstractmethod
    def _evaluate(self, *args, **kwargs):
        """Evaluate the output of the envelope.

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
        t: Array
            One-dimensional vector of timestamps.

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


class ConstantEnvelope(Envelope):
    """A constant envelope tone with a fixed length.

    _amplitude: Quantity
        The amplitude of the envelope.
    _t_final: Quantity
        The length in time of the envelope.
    _gradientFunction: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _grad_arg_nums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.

    """

    @partial(jit, static_argnums=(0,))
    def _evaluate(
        self,
        amp: Array,
        t_final: Array,
        t: Array | float,
    ) -> Array:
        """Evaluate the envelope depending on all parameters.

        Abstract method.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        return jnp.squeeze(jnp.where(t <= t_final, amp, 0.0))

    def compute_output(self, t: Array) -> Array:
        """Compute the constant signal envelope at different times.

        Parameters
        ----------
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Output of the computation.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        amp = self.amplitude.get_value()
        t_final = self.t_final.get_value()
        return self._evaluate(amp, t_final, t)  # type: ignore

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
        return jnp.zeros_like(t)


class ZeroEnvelope(ConstantEnvelope):
    """Shorthand implentation of a zero signal envelope.

    _amplitude: Quantity
        The amplitude of the envelope.
    _t_final: Quantity
        The length in time of the envelope.
    _gradientFunction: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _grad_arg_nums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.

    """

    def __init__(self):
        super().__init__()
        self.amplitude.set_value(0.0)


class FlatTopGaussianEnvelope(Envelope):
    """A flat-top Gaussian envelope.

    _amplitude: Quantity
        The amplitude of the envelope.
    _t_final: Quantity
        The length in time of the envelope.
    _gradient_function: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _grad_arg_nums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.

    """

    @partial(jit, static_argnums=(0,))
    def _evaluate(self, amp: Array, t_final: Array, t: Array | float):  # type: ignore
        """Compute the output of the device.

        Explicitly depends on the optimisable parameters.

        Parameters
        ----------
        amp: Quantity
            Cosine pulse amplitude.
        t_final: Array
            The length in time of the entire envelope.
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the output of the device that explicitly depends
            on the optimisable parameters.

        """
        ramp_time = t_final / 10
        ramp_up = 1 + erf((t - t_final / 5) / ramp_time)
        ramp_down = 1 + erf((-t + 4 * t_final / 5) / ramp_time)
        return amp * ramp_up * ramp_down / 4

    @staticmethod
    @jit
    def __dir_erf(x: Array):
        return 2 / jnp.sqrt(jnp.pi) * jnp.exp(-(x**2))

    @partial(jit, static_argnums=(0,))
    def _evaluate_time_grad(self, amp: Array, t_final: Array, t: Array):
        """Compute the output of the device.

        Explicitly depends on the optimisable parameters.

        Parameters
        ----------
        amp: Quantity
            Cosine pulse amplitude.
        t_final: Array
            The length in time of the entire envelope.
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the output of the device that explicitly depends
            on the optimisable parameters.

        """
        ramp_time = t_final / 10

        ramp_up = 1 + erf((t - t_final / 5) / ramp_time)
        ramp_up_t_dir = self.__dir_erf((t - t_final / 5) / ramp_time)
        ramp_up_t_dir /= ramp_time

        ramp_down = 1 + erf((-t + 4 * t_final / 5) / ramp_time)
        ramp_down_t_dir = self.__dir_erf((-t + 4 * t_final / 5) / ramp_time)
        ramp_down_t_dir *= -1 / ramp_time

        prod_dir = ramp_up * ramp_down_t_dir + ramp_up_t_dir * ramp_down

        return amp * prod_dir / 4

    @partial(jit, static_argnums=(0,))
    def _evaluate_t_final_grad(self, amp: Array, t_final: Array, t: Array):
        """Compute the output of the device.

        Explicitly depends on the optimisable parameters.

        Parameters
        ----------
        amp: Quantity
            Cosine pulse amplitude.
        t_final: Array
            The length in time of the entire envelope.
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the output of the device that explicitly depends
            on the optimisable parameters.

        """
        ramp_time = t_final / 10

        ramp_up = 1 + erf((t - t_final / 5) / ramp_time)
        ramp_up_t_fin_dir = self.__dir_erf((t - t_final / 5) / ramp_time)
        ramp_up_t_fin_dir *= -1 / (5 * ramp_time)

        ramp_down = 1 + erf((-t + 4 * t_final / 5) / ramp_time)
        ramp_down_t_fin_dir = self.__dir_erf((-t + 4 * t_final / 5) / ramp_time)
        ramp_down_t_fin_dir *= 4 / (5 * ramp_time)

        prod_dir = ramp_up * ramp_down_t_fin_dir + ramp_up_t_fin_dir * ramp_down

        return amp * prod_dir / 4

    def compute_output(self, t: Array) -> Array:
        """Get the output of the device on time stamps.

        Parameters
        ----------
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the output of the device.

        """
        amp = self.amplitude.get_value()
        t_final = self.t_final.get_value()
        return self._evaluate(amp, t_final, t)  # type: ignore

    def compute_gradient(self, t: Array) -> Array:
        """Return the gradient wrt dimensionless parameters.

        Parameters
        ----------
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns the gradient wrt dimensionless parameters.

        """
        amp = self.amplitude.get_value()
        t_final = self.t_final.get_value()
        t_arr = jnp.array(t, ndmin=1)

        grads = []
        if self._is_optimised(self.amplitude):
            grads.append(self._evaluate(jnp.array([1.0]), t_final, t_arr))
        if self._is_optimised(self.t_final):
            grads.append(self._evaluate_t_final_grad(amp, t_final, t_arr))
        return jnp.stack(grads, axis=1) if len(grads) > 0 else jnp.empty((t_arr.shape[0], 0))

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
        amp = self.amplitude.get_value()
        t_final = self.t_final.get_value()
        return jnp.array(self._evaluate_time_grad(amp, t_final, t))


class GaussEnvelope(Envelope):
    """Create a simple Gauss envelope.

    _amplitude: Quantity
        The amplitude of the envelope.
    _t_final: Quantity
        The length in time of the envelope.
    _gradient_function: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _grad_arg_nums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.

    """

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate(self, amp: Array, t_final: Array, t: Array) -> Array:  # type: ignore
        """Calculate the unscaled gaussian signal.

        Parameters
        ----------
        t_final: Array
            Duration of the signal to calculate the center of the gaussian from.
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            The unscaled gaussian signal.
        """
        sigma = t_final / 8
        env = amp * jnp.exp(-(1 / 2) * (t - t_final / 2) ** 2 / sigma**2)
        return jnp.squeeze(env)

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate_time_gradient(self, amp: Array, t_final: Array, t: Array) -> Array:
        """Calculate the unscaled gaussian signal.

        Parameters
        ----------
        t_final: Array
            Duration of the signal to calculate the center of the gaussian from.
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            The unscaled gaussian signals time derivative.
        """
        sigma = t_final / 8
        time_grad = self._evaluate(amp, t_final, t) * -1.0 * (t - t_final / 2) / sigma**2
        return time_grad  # type: ignore

    def compute_output(self, t: Array) -> Array:
        """Compute a Gaussian signal.

        Parameters
        ----------
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns a vector gaussian signal.
        """
        t_final = self.t_final.get_value()
        amp = self.amplitude.get_value()
        return self._evaluate(amp, t_final, t)  # type: ignore

    def compute_time_gradient(self, t: Array) -> Array:
        """Compute a Gaussian signals time derivative.

        Parameters
        ----------
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns a vector gaussian signals time derivative.
        """
        t_final = self.t_final.get_value()
        amp = self.amplitude.get_value()
        env_time_deriv = self._evaluate_time_gradient(amp, t_final, t)
        return env_time_deriv  # type: ignore


class dCRABEnvelope(Envelope):
    r"""Create a dCRAB pulse envelope.

    The dCRAB pulse is given as a sum of sinusoidal components as [Müller2022]
    $$f(t) = g(t)\Big( 1 + \sum_{i=1}^{N_c / 2} c_{2i} \frac{\cos(\omega_{2i} t)}{\Lambda(t)}
        + \sum_{i = 1} ^ {N_c/2} c_{2i + 1} \frac{sin(\omega_{2i + 1} t)}{\Lambda(t)} \Big)$$

    Here we consider $g(t) = \Lambda(t) = 1$ for simplicity.
    Further, even components are for cosine and odd components are for sine.
    *Note - The function is designed to work well for even total number of components.
    For odd total number it may not work as expected.*

    [Müller2022] Müller et al. "One decade of quantum optimal control in the chopped random basis"

    _amplitude: Quantity
        The amplitude of the envelope.
    _t_final: Quantity
        The length in time of the envelope.
    _num_components: int
        Number of components added each iteration to the dCRAB basis. Defaults to 2. Adviced to be an even number.
    _total_num_components: int
        Total number of components in the current dCRAB basis. This is the number of coefficients
        or the number of frequencies present. NOT the sum of them.
    _coefficients: list[Quantity]
        Vector quantity as a list of amplitudes of individual sinusoidal components.
    _frequencies: list[Quantity]
        Vector quantity as a list of frequencies of individual sinusoidal components.
    _gradient_function: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _grad_arg_nums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.
    """

    _amplitude: Quantity
    _t_final: Quantity
    _num_components: int
    _total_num_components: int
    _all_coefficients: list[Quantity]
    _all_frequencies: list[Quantity]
    __min_frequency: float
    __max_frequency: float

    def __init__(
        self,
        amplitude: Quantity | None = None,
        t_final: Quantity | None = None,
        num_components: int = 2,
        min_frequency: float = 0.0,
        max_frequency: float = 2 * jnp.pi * 5.0,
        seeds: tuple[int, int] | None = None,
    ):
        self._amplitude = amplitude or Quantity(
            1.55e8,
            min_value=jnp.array(0.0),
            max_value=jnp.array(1e9),
            unit="Hz",
            name="Amplitude",
            two_pi=True,
        )

        self._t_final = t_final or Quantity(
            32e-9,
            min_value=jnp.array(0),
            max_value=jnp.array(100e-9),
            unit="s",
            name="t_final",
        )

        self._num_components = num_components

        if seeds is None:
            seed = int(time.time() + 10)  # to make sure the seeds are different for the two cases
        else:
            seed = seeds[0]

        key = jax.random.key(seed)
        coeffs = jax.random.uniform(key, shape=(self._num_components,), minval=0, maxval=1)

        self._all_coefficients = [
            Quantity(
                coeffs[i],
                min_value=jnp.array(0.0),
                max_value=jnp.array(1.0),
                unit="",
                name=f"CRAB coefficient {i}",
            )
            for i in range(self._num_components)
        ]

        if seeds is None:
            seed = int(time.time() + 10)  # to make sure the seeds are different for the two cases
        else:
            seed = seeds[1]

        key = jax.random.key(seed)
        self.__min_frequency = min_frequency
        self.__max_frequency = max_frequency
        freqs = jax.random.uniform(
            key, shape=(self._num_components,), minval=self.__min_frequency, maxval=self.__max_frequency
        )

        self._all_frequencies = [
            Quantity(
                freqs[i],
                min_value=jnp.array(self.__min_frequency),
                max_value=jnp.array(self.__max_frequency),
                unit="Hz",
                name=f"CRAB frequency {i}",
                two_pi=True,
            )
            for i in range(self._num_components)
        ]

        self._total_num_components = self._num_components

        self._gradient_function: Callable | None = None
        self._grad_arg_nums: tuple[int, ...] = ()

    def get_parameters(self):
        """Return the parameters of the CRAB signal.
        The parameters are arranged as follows,
            [amplitude, t_final, ... total_num coefficients ..., ... total_num frequencies ...]
        """
        params = [self.amplitude, self._t_final]
        params.extend(self._all_coefficients)
        params.extend(self._all_frequencies)
        return params

    def add_new_components(self, seeds: tuple[int, int] | None = None):
        """Add `self._num_components` number of new randomized components to the optimization."""
        if seeds is None:
            seed = int(time.time())
        else:
            seed = seeds[0]

        key = jax.random.key(seed)
        coeffs = jax.random.uniform(key, shape=(self._num_components,), minval=0, maxval=1)

        self._all_coefficients.extend(
            [
                Quantity(
                    coeffs[i],
                    min_value=jnp.array(0.0),
                    max_value=jnp.array(1.0),
                    unit="",
                    name=f"CRAB coefficient {i + self._total_num_components}",
                )
                for i in range(self._num_components)
            ]
        )

        if seeds is None:
            seed = int(time.time())
        else:
            seed = seeds[1]

        key = jax.random.key(seed)
        freqs = jax.random.uniform(
            key, shape=(self._num_components,), minval=self.__min_frequency, maxval=self.__max_frequency
        )

        self._all_frequencies.extend(
            [
                Quantity(
                    freqs[i],
                    min_value=jnp.array(0.0),
                    max_value=jnp.array(2 * jnp.pi * 5.0),
                    unit="Hz",
                    name=f"CRAB frequency {i + self._total_num_components}",
                    two_pi=True,
                )
                for i in range(self._num_components)
            ]
        )

        self._total_num_components += self._num_components

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate(self, *params: Array) -> Array:  # type: ignore
        """Compute the CRAB pulse.

        Here the params is arranged as follows,
            [amplitude, t_final, ... total_num coefficients ..., ... total_num frequencies ..., t]

        This evaluate function is written in this way to make it compatible with adding new
        components and freezing existing components required for dCRAB optimisation.
        """
        amp = params[0]
        t_final = params[1]
        coeffs: list[Array] = params[2 : 2 + self._total_num_components]  # type: ignore
        freqs: list[Array] = params[2 + self._total_num_components : -1]  # type: ignore
        t = params[-1]
        env = jnp.zeros_like(t)
        for i in range(int(self._total_num_components / 2)):
            env += coeffs[2 * i] * jnp.cos(freqs[2 * i] * t / t_final)
            env += coeffs[2 * i + 1] * jnp.sin(freqs[2 * i + 1] * t / t_final)
        env /= 2 * jnp.sum(jnp.array(coeffs))
        return jnp.squeeze(amp * env)

    def compute_output(self, t: Array) -> Array:
        """Compute the CRAB signal.

        Parameters
        ----------
        t: Array
            One-dimensional vector of timestamps.

        Returns
        -------
        Array
            Returns a vector CRAB signal.
        """
        t_final = self.t_final.get_value()
        amp = self.amplitude.get_value()
        coeffs = [coeff.get_value() for coeff in self._all_coefficients]
        freqs = [freq.get_value() for freq in self._all_frequencies]
        params = [amp, t_final] + coeffs + freqs
        return self._evaluate(*params, t)  # type: ignore
