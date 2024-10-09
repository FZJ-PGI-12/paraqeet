"""Class definition for the Evelopes."""

from abc import abstractmethod
from collections.abc import Callable
from functools import partial
import numpy as np

import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from jax.scipy.special import erf
from jax.typing import ArrayLike as Array

from cthree.Optimisable import Optimisable
from cthree.Quantity import Quantity

jax.config.update("jax_enable_x64", True)


class Envelope(Optimisable):
    """Classical Signal Envelope class.

    __amplitude: Quantity
        The amplitude of the envelope.
    __t_final: Quantity
        The length in time of the envelope.
    _gradientFunction: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _gradArgNums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.

    """

    def __init__(
        self,
        amplitude: Quantity | None = None,
        t_final: Quantity | None = None,
    ):
        self.__amplitude = amplitude or Quantity(
            2e5 * 2 * np.pi,
            min_value=np.array(1e5 * 2 * np.pi),
            max_value=np.array(250e6 * 2 * np.pi),
            unit="Hz",
            name="Amplitude",
        )

        self.__t_final = t_final or Quantity(
            32e-9,
            min_value=np.array(0e-9),
            max_value=np.array(100e-9),
            unit="s",
            name="t_final",
        )

        self._gradientFunction: Callable | None = None
        self._gradArgNums: tuple[int, ...] = ()

    def getParameters(self):
        """Get a list of parameters of the envelope.

        Returns
        -------
        List[Quantity]
            List of parameters of the envelope.

        """
        return [self.__amplitude, self.t_final]

    @property
    def amplitude(self) -> Quantity:
        """Get the amplitude of the system.

        Returns
        -------
        cthree.Quantity
            Amplitude of the system.

        """
        return self.__amplitude

    @amplitude.setter
    def amplitude(self, amplitude: Quantity) -> None:
        """Set the amplitude of the system.

        Parameters
        ----------
        cthree.Quantity
            Amplitude value of the system to be set.

        """
        self.__amplitude = amplitude

    @property
    def t_final(self) -> Quantity:
        """Get the length of the tone.

        Returns
        -------
        cthree.Quantity
            Length in time of the tone.

        """
        return self.__t_final

    @t_final.setter
    def t_final(self, t_final: Quantity) -> None:
        """Set the length of the tone.

        Parameters
        ----------
        cthree.Quantity
            Length in time of the tone to be set.

        """
        self.__t_final = t_final

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
        """Evaluate the output of the envelope.

        Abstract method.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        raise NotImplementedError()

    @abstractmethod
    def computeOutput(
        self, t: np.ndarray, t_start: np.ndarray = 0.0
    ) -> np.ndarray:
        """Compute the output.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.
        t_start: np.ndarray
            The timestamp at which to start the signal.

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

    def computeGradient(
        self, t: np.ndarray, t_start: np.ndarray = 0.0
    ) -> Array:
        """Compute the gradient of the `_evaluate` method.

        Uses Automatic differentiation.
        The `_evaluate` method should be a `pure` function (should take the
        optimisable parameters as function arguments and doesn't depend on
        global variables).
        Refer to https://jax.readthedocs.io/en/latest/notebooks/
        Common_Gotchas_in_JAX.html for functionally `pure` functions.
        To implement analytical gradients / other methods for gradient
        computation overwrite this method in the inherited class.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.
        t_start: np.ndarray
            The timestamp at which to start the signal.

        Returns
        -------
        numpy.ndarray
            Returns the gradient array of the `_evaluate` method.

        """
        params = self.getParameters()
        param_values = [param.getValue() for param in params]
        t = jnp.array(t - t_start, ndmin=1)

        grads = jnp.empty((t.shape[0], 0))

        if self._gradientFunction is not None:
            grads = jnp.stack(self._gradientFunction(*param_values, t), axis=1)

            parameter_scales = jnp.array(
                [param.getScale() for param in self._optimisableParameters]
            )

            if len(self._gradArgNums) > 1:
                parameter_scales = jnp.reshape(
                    parameter_scales, (1,) + parameter_scales.shape
                )
                grads = jnp.squeeze(grads) * parameter_scales
            else:
                grads = jnp.squeeze(grads) * parameter_scales
                grads = jnp.reshape(grads, (-1, 1))

        return grads

    def computeTimeGradient(
        self, t: np.ndarray, t_start: np.ndarray = 0.0
    ) -> Array:
        """Compute a signal envelopes time derivative.

        Parameters
        ----------
        t: np.ndarray
            One-dimensional vector of timestamps.
        t_start: np.ndarray
            The timestamp at which to start the signal.

        Returns
        -------
        np.ndarray
            Returns a vector signals time derivative.
        """
        t = jnp.array(t - t_start, ndmin=1)
        envTimeGradFun = grad(self.computeOutput, argnums=0)
        envTimeGrad = vmap(envTimeGradFun, in_axes=(0,))(t)
        return jnp.squeeze(envTimeGrad)


class ConstantEnvelope(Envelope):
    """A constant envelope tone with a fixed length.

    __amplitude: Quantity
        The amplitude of the envelope.
    __t_final: Quantity
        The length in time of the envelope.
    _gradientFunction: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _gradArgNums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.

    """

    def _evaluate(
        self,
        amp: np.ndarray,
        t_final: np.ndarray,
        t: np.ndarray,
        t_start: np.ndarray = 0.0,
    ) -> Array:
        """Evaluate the envelope depending on all parameters.

        Abstract method.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        end_time = t_start + t_final
        return jnp.array(
            [amp if t_start <= time <= end_time else 0.0 for time in t]
        )

    def computeOutput(self, t: np.ndarray, t_start: np.ndarray = 0.0) -> Array:
        """Compute the constant signal envelope at different times.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.
        t_start: np.ndarray
            The timestamp at which to start the signal.

        Returns
        -------
        numpy.ndarray
            Output of the computation.

        Raises
        ------
        NotImplementedError
            Subclasses derived from this class must implement this method.

        """
        amp = self.__amplitude.getValue()
        t_final = self.t_final.getValue()
        return self._evaluate(amp, t_final, t, t_start)

    def computeTimeGradient(
        self, t: np.ndarray, t_start: np.ndarray = 0.0
    ) -> Array:
        """Compute a signal envelopes time derivative.

        Parameters
        ----------
        t: np.ndarray
            One-dimensional vector of timestamps.
        t_start: np.ndarray
            The timestamp at which to start the signal.

        Returns
        -------
        np.ndarray
            Returns a vector signals time derivative.
        """
        return jnp.zeros_like(t)


class ErfEnvelope(Envelope):
    """An error-function shaped envelope.

    __amplitude: Quantity
        The amplitude of the envelope.
    __t_final: Quantity
        The length in time of the envelope.
    _gradientFunction: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _gradArgNums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.

    """

    @partial(jit, static_argnums=(0,))
    def _evaluate(self, amp: np.ndarray, t_final: np.ndarray, t: np.ndarray):
        """Compute the output of the device.

        Explicitly depends on the optimisable parameters.

        Parameters
        ----------
        amp : cthree.Quantity
            Cosine pulse amplitude.
        t_final: np.ndarray
            The length in time of the entire envelope.
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        jax.numpy.ndarray
            Returns the output of the device that explicitly depends
            on the optimisable parameters.

        """
        ramp_time = t_final / 10
        rampUp = 1 + erf((t - t_final / 5) / ramp_time)
        rampDown = 1 + erf((-t + 4 * t_final / 5) / ramp_time)
        return amp * rampUp * rampDown / 4

    @staticmethod
    @jit
    def __dir_erf(x: np.ndarray):
        return 2 / np.sqrt(np.pi) * np.exp(-(x**2))

    @partial(jit, static_argnums=(0,))
    def _evaluateTimeGrad(
        self, amp: np.ndarray, t_final: np.ndarray, t: np.ndarray
    ):
        """Compute the output of the device.

        Explicitly depends on the optimisable parameters.

        Parameters
        ----------
        amp : cthree.Quantity
            Cosine pulse amplitude.
        t_final: np.ndarray
            The length in time of the entire envelope.
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        jax.numpy.ndarray
            Returns the output of the device that explicitly depends
            on the optimisable parameters.

        """
        ramp_time = t_final / 10

        rampUp = 1 + erf((t - t_final / 5) / ramp_time)
        rampUp_t_dir = self.__dir_erf((t - t_final / 5) / ramp_time)
        rampUp_t_dir /= ramp_time

        rampDown = 1 + erf((-t + 4 * t_final / 5) / ramp_time)
        rampDown_t_dir = self.__dir_erf((-t + 4 * t_final / 5) / ramp_time)
        rampDown_t_dir *= -1 / ramp_time

        prod_dir = rampUp * rampDown_t_dir + rampUp_t_dir * rampDown

        return amp * prod_dir / 4

    @partial(jit, static_argnums=(0,))
    def _evaluateTFinalGrad(
        self, amp: np.ndarray, t_final: np.ndarray, t: np.ndarray
    ):
        """Compute the output of the device.

        Explicitly depends on the optimisable parameters.

        Parameters
        ----------
        amp : cthree.Quantity
            Cosine pulse amplitude.
        t_final: np.ndarray
            The length in time of the entire envelope.
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        jax.numpy.ndarray
            Returns the output of the device that explicitly depends
            on the optimisable parameters.

        """
        ramp_time = t_final / 10

        rampUp = 1 + erf((t - t_final / 5) / ramp_time)
        rampUp_t_fin_dir = self.__dir_erf((t - t_final / 5) / ramp_time)
        rampUp_t_fin_dir *= -1 / (5 * ramp_time)

        rampDown = 1 + erf((-t + 4 * t_final / 5) / ramp_time)
        rampDown_t_fin_dir = self.__dir_erf((-t + 4 * t_final / 5) / ramp_time)
        rampDown_t_fin_dir *= 4 / (5 * ramp_time)

        prod_dir = rampUp * rampDown_t_fin_dir + rampUp_t_fin_dir * rampDown

        return amp * prod_dir / 4

    def computeOutput(self, t: np.ndarray, t_start: np.ndarray = 0.0) -> Array:
        """Get the output of the device on time stamps.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.
        t_start: np.ndarray
            The timestamp at which to start the signal.

        Returns
        -------
        numpy.ndarray
            Returns the output of the device.

        """
        amp = self.__amplitude.getValue()
        t_final = self.t_final.getValue()
        return self._evaluate(amp, t_final, t - t_start)

    def computeGradient(
        self, t: np.ndarray, t_start: np.ndarray = 0.0
    ) -> Array:
        """Return the gradient wrt dimensionless parameters.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensional vector of timestamps.
        t_start: np.ndarray
            The timestamp at which to start the signal.

        Returns
        -------
        numpy.ndarray
            Returns the gradient wrt dimensionless parameters.

        """
        amp = self.__amplitude.getValue()
        t_final = self.t_final.getValue()
        t = np.array(t - t_start, ndmin=1)

        grads = []
        if self._isOptimised(self.__amplitude):
            dc_dAmp = self._evaluate(np.array(1.0), t_final, t)
            grads.append(self.__amplitude.getScale() * dc_dAmp)
        if self._isOptimised(self.t_final):
            dc_tFinal = self._evaluateTFinalGrad(amp, t_final, t)
            grads.append(self.t_final.getScale() * dc_tFinal)
        return (
            jnp.stack(grads, axis=1)
            if len(grads) > 0
            else jnp.empty((t.shape[0], 0))
        )

    def computeTimeGradient(
        self, t: np.ndarray, t_start: np.ndarray = 0.0
    ) -> Array:
        """Compute a signal envelopes time derivative.

        Parameters
        ----------
        t: np.ndarray
            One-dimensional vector of timestamps.
        t_start: np.ndarray
            The timestamp at which to start the signal.

        Returns
        -------
        np.ndarray
            Returns a vector signals time derivative.
        """
        amp = self.__amplitude.getValue()
        t_final = self.t_final.getValue()
        return self._evaluateTimeGrad(amp, t_final, t - t_start)


class GaussEnvelope(Envelope):
    """Create a simple Gauss envelope.

    __amplitude: Quantity
        The amplitude of the envelope.
    __t_final: Quantity
        The length in time of the envelope.
    _gradientFunction: Callable | None
        The function to calculate the gradient with respect to a set of
        previously defined parameters.
    _gradArgNums: tuple[int, ...]
        The identifying indices of which parameters to calculate the gradient
        with respect to.

    """

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate(
        self, amp: np.ndarray, t_final: np.ndarray, t: np.ndarray
    ) -> Array:
        """Calculate the unscaled gaussian signal.

        Parameters
        ----------
        t_final : np.ndarray
            Duration of the signal to calculate the center of the gaussian from.
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            The unscaled gaussian signal.
        """
        sigma = t_final / 8
        env = amp * jnp.exp(-(1 / 2) * (t - t_final / 2) ** 2 / sigma**2)
        return jnp.squeeze(env)

    @partial(jax.jit, static_argnums=(0,))
    def _evaluateTimeGradient(
        self, amp: np.ndarray, t_final: np.ndarray, t: np.ndarray
    ) -> Array:
        """Calculate the unscaled gaussian signal.

        Parameters
        ----------
        t_final : np.ndarray
            Duration of the signal to calculate the center of the gaussian from.
        t : numpy.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        numpy.ndarray
            The unscaled gaussian signals time derivative.
        """
        sigma = t_final / 8
        timeGrad = (
            self._evaluate(amp, t_final, t)
            * -1.0
            * (t - t_final / 2)
            / sigma**2
        )
        return jnp.squeeze(timeGrad)

    def computeOutput(self, t: np.ndarray, t_start: np.ndarray = 0.0) -> Array:
        """Compute a Gaussian signal.

        Parameters
        ----------
        t: np.ndarray
            One-dimensional vector of timestamps.
        t_start: np.ndarray
            The timestamp at which to start the signal.

        Returns
        -------
        np.ndarray
            Returns a vector gaussian signal.
        """
        t_final = self.t_final.getValue()
        amp = self.__amplitude.getValue()
        return self._evaluate(amp, t_final, t - t_start)

    def computeTimeGradient(
        self, t: np.ndarray, t_start: np.ndarray = 0.0
    ) -> Array:
        """Compute a Gaussian signals time derivative.

        Parameters
        ----------
        t: np.ndarray
            One-dimensional vector of timestamps.
        t_start: np.ndarray
            The timestamp at which to start the signal.

        Returns
        -------
        np.ndarray
            Returns a vector gaussian signals time derivative.
        """
        t_final = self.t_final.getValue()
        amp = self.__amplitude.getValue()
        envTimeDeriv = self._evaluateTimeGradient(amp, t_final, t - t_start)
        return envTimeDeriv
