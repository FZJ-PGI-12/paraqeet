"""Class definition for the Device model."""

from abc import abstractmethod
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap, jit
from jax.typing import ArrayLike as Array

from cthree.Optimisable import Optimisable
from cthree.Quantity import Quantity

jax.config.update("jax_enable_x64", True)


class Waveform(Optimisable):
    """Classical electronics."""

    _gradientFunction: Callable | None = None
    _gradArgNums: tuple[int, ...] = ()

    def _computeGradientFunction(
        self,
        signalFunction: Callable,
        argnums: tuple[int, ...],
        vmap_axes: tuple[int, ...],
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
        self._gradientFunction = jit(partial_grads)

    def setOptimisableParameters(self, params: list[Quantity]) -> None:
        """Set optimisable parameters for optimisation.

        Parameters
        ----------
        params : List[cthree.Quantity]
            Input list of parameters to be set.

        """
        super().setOptimisableParameters(params)

        self._gradArgNums = ()
        for i, param in enumerate(self.getParameters()):
            if self._isOptimised(param):
                self._gradArgNums += (i,)

        # Recompute gradient function
        params = self.getParameters()
        num_params = len(params)

        # vmap over time axis only, set everything else to None
        vmap_axes = (None,) * num_params
        vmap_axes += (0,)  # type: ignore

        if len(self._gradArgNums) > 0:
            self._computeGradientFunction(
                self._evaluate,
                argnums=self._gradArgNums,
                vmap_axes=vmap_axes,
            )
        else:
            self._gradientFunction = None

    @abstractmethod
    def _evaluate(self, *args, **kwargs):
        """Evaluate the output of the system.

        Abstract method.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    @abstractmethod
    def computeOutput(self, t: np.ndarray) -> np.ndarray:
        """Compute the output.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            Output of the computation.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    def computeGradient(self, t: np.ndarray) -> np.ndarray:
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
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            Returns the gradient array of the `_evaluate` method.

        """
        params = self.getParameters()
        param_values = [param.getValue() for param in params]
        t = jnp.array(t, ndmin=1)
        grads = jnp.empty((t.shape[0], 0))
        if self._gradientFunction is not None:
            grads = jnp.stack(self._gradientFunction(*param_values, t), axis=1)
            grads = jnp.squeeze(grads, -1)
        return grads

    def computeTimeGradient(self, t: np.ndarray) -> Array:
        """Compute a signal envelopes time derivative.

        Parameters
        ----------
        t: np.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        np.ndarray
            Returns a vector signals time derivative.

        """
        t = jnp.array(t, ndmin=1)
        envTimeGradFun = grad(self.computeOutput, argnums=0)
        envTimeGrad = vmap(envTimeGradFun, in_axes=(0,))(t)
        return jnp.squeeze(envTimeGrad)


class LocalOscillator(Waveform):
    """A local oscillators carrier signal.

    __lo_freq : Quantity
        The frequency of the carrier signal.

    """

    __lo_freq: Quantity

    def __init__(self, frequency: Quantity | None = None) -> None:
        self.__lo_freq = frequency or Quantity(
            value=np.array(4.8e9 * 2 * np.pi),
            min_value=np.array(0.8 * 4.8e9 * 2 * np.pi),
            max_value=np.array(1.2 * 4.8e9 * 2 * np.pi),
            unit="Hz",
            name="lo_freq",
            twoPi=True,
        )

    def getParameters(self) -> list[Quantity]:
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
    def _evaluate(self, freq: np.ndarray, t: np.ndarray) -> Array:
        """Calculate the unscaled carrier signal.

        Parameters
        ----------
        freq : numpy.ndarray
            The frequency of the carrier signal
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            The unscaled the carrier signal.
        """
        return jnp.exp(1j * freq * t)

    def computeOutput(self, t: np.ndarray) -> Array:
        """Evaluate a carrier signal from an input time vector.

        Parameters
        ----------
        t : np.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        np.ndarray
            Returns a vector carrier signal.
        """
        return self._evaluate(self.__lo_freq.getValue(), t)

    def computeGradient(self, t: np.ndarray) -> Array:
        """Return the gradient wrt to frequency of carrier signal.

        Parameters
        ----------
        t : np.ndarray
            Array of time points to evaluate gradients at.

        Returns
        -------
        np.ndarray
            Gradient of tone wrt to frequency.
        """
        freq = self.__lo_freq.getValue()
        t = jnp.array(t, ndmin=1)

        grads = jnp.empty((t.shape[0], 0))
        if self._isOptimised(self.__lo_freq):
            grads = jnp.reshape(1j * t * self._evaluate(freq, t), (-1, 1))

        return grads

    def computeTimeGradient(self, t: np.ndarray) -> Array:
        """Compute a signals time derivative.

        Parameters
        ----------
        t: np.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        np.ndarray
            Returns a vector signals time derivative.

        """
        freq = self.__lo_freq.getValue()
        return 1j * t * self._evaluate(freq, t)


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

    def __init__(
        self,
        envelopes: Waveform | list[Waveform],
        deltas: list[Quantity] | None = None,
    ) -> None:
        self.__envs = envelopes if isinstance(envelopes, list) else [envelopes]
        self.__add_deltas(self.__envs, deltas)

    def getParameters(self) -> list[Quantity]:
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
            params += tone.getParameters()
            params += [self.__getToneDelta(tone)]
        return params

    @staticmethod
    def __add_deltas(envelopeTones: list[Waveform], deltas: list[Quantity]) -> None:
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
                    deltas[ii] if deltas else np.array(-200e6 * 2 * np.pi),
                    min_value=np.array(-3 * 200e6 * 2 * np.pi),
                    max_value=np.array(-0.1 * 200e6 * 2 * np.pi),
                    unit="Hz",
                    name="Delta",
                ),
            )

    @staticmethod
    def __getToneDelta(tone: Waveform) -> Quantity:
        """Return a list of deltas for each tone.

        Returns
        -------
        Quantity
            List of delta values for each tone.
        """
        return tone.__getattribute__("_" + tone.__class__.__name__ + "__delta")

    def _evaluate(self, t, *deltas) -> Array:
        """Compute the DRAG Envelope using deltas.

        Explicit function depending on deltas to compute gradients using AD.

        Parameters
        ----------
        t : np.ndarray
            One-dimensional vector of timestamps.
        deltas: List[float]
            Variable number of inputs for delta parameters for each tone.

        Returns
        -------
        numpy.ndarray
            Returns a vector signal of the DRAG envelope.
        """
        total_env = jnp.zeros_like(t, dtype=np.complex128)
        for delta, tone in zip(deltas, self.__envs):
            env = tone.computeOutput(t)
            env_grad = tone.computeTimeGradient(t)
            total_env += env - 1.0j / delta * env_grad
        return jnp.squeeze(total_env)

    def computeOutput(self, t: np.ndarray) -> Array:
        """Evaluate a carrier signal from an input time vector.

        Parameters
        ----------
        t : np.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        np.ndarray
            Returns a vector carrier signal.
        """
        deltas = [self.__getToneDelta(tone).getValue() for tone in self.__envs]
        return self._evaluate(t, *deltas)

    def setOptimisableParameters(self, params: list[Quantity]) -> None:
        """Set specified parameters to be optimised.

        Also add the indices to `__gradArgNums` to compute the gradients.

        Parameters
        ----------
        params : list[Quantity]
        """
        super().setOptimisableParameters(params)

        for tone in self.__envs:
            tone.setOptimisableParameters(params)

    def computeGradient(self, t: np.ndarray) -> Array:
        """Generate gradient of the signal for an array of time.

        Collect and return the parameter gradients from the Tone and the carrier
        Tone. Compute the gradient of the generator parameters by AD.
        The order of the gradients should match the order of paramters in
        `self.getParameter()` method

        Parameters
        ----------
        t : np.ndarray
            An array of time points.

        Returns
        -------
        jnp.ndarray
            Array of gradients wrt each parameter for each time point.
        """
        deltas = [self.__getToneDelta(tone) for tone in self.__envs]
        delta_values = [delta.getValue() for delta in deltas]

        gradients = jnp.zeros(shape=(t.shape[0], 0))

        # Collect gradients wrt envelope parameters
        for tone in self.__envs:
            grads = tone.computeGradient(t)
            gradients = jnp.append(gradients, grads, axis=1)

        # Collect gradients wrt deltas
        for i, tone in enumerate(self.__envs):
            if self._isOptimised(deltas[i]):
                grad = (
                    1j * (delta_values[i] ** 2) * tone.computeTimeGradient(t)
                    # * delta_scales[i]
                )
                grad = jnp.expand_dims(grad, axis=1)
                gradients = jnp.append(gradients, grad, axis=1)

        gradients = jnp.append(gradients, grads, axis=1)

        return jnp.array(gradients)
