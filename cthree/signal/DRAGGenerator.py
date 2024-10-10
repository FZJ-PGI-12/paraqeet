"""Generator to produce pulses with DRAG corrections.

Adds `delta` and `phase` parameters to the pulse as optimisable parameters.
"""

from collections.abc import Callable

import numpy as np
from jax import vmap, grad
from jax import numpy as jnp

from cthree.Quantity import Quantity
from cthree.signal.Waveform import Waveform, LocalOscillator
from cthree.signal.Generator import Generator

import jax

jax.config.update("jax_enable_x64", True)


class DRAGGenerator(Generator):
    """Create a DRAG corrected signal from an envelope Tone.

    __phase:
        The phase of the signal to get direction in xy-plane.
    __envelopeTone:
        The tone that defines the envelope of the DRAG corrected signal.
    __carrierTone:
        Fast oscillating part of the DRAG corrected signal.
    __gradientFunction:
        The function to evaluate the DRAG corrected signal's gradient from.
    __gradArgNums:
        The arguments passed to the gradient function.
    __optimisableParams:
        List of generator parameters (phase and deltas) to be optimised.
    """

    __envelopeTones: list[Waveform]
    __carrierTone: LocalOscillator
    __phase: Quantity
    __gradientFunction: Callable = None
    __gradArgNums: tuple[int, ...] = ()
    __optimisableParams: list[Quantity] = []

    def __init__(
        self,
        devices: list[Waveform] | None,
        carrier_freq: Quantity | None = None,
        phase: Quantity | None = None,
        deltas: list[Quantity] | None = None,
    ):
        self.__envelopeTones = self.__add_deltas(devices or [], deltas)
        self.__carrierTone = LocalOscillator(frequency=carrier_freq)

        self.__phase = phase or Quantity(
            np.array(0.0),
            min_value=np.array(-np.pi),
            max_value=np.array(np.pi),
            unit="rad",
            name="Phase",
        )

    @property
    def carrier_frequency(self) -> Quantity:
        """Get the signal carrier's oscillation frequency.

        Returns
        -------
        Quantity
            The signal carrier's frequency.
        """
        return self.__carrierTone.frequency

    @carrier_frequency.setter
    def carrier_frequency(self, freq: Quantity) -> None:
        """Set the signal carrier's oscillation frequency.

        Parameters
        ----------
        freq: Quantity
            The signal carrier's frequency.
        """
        self.__carrierTone.frequency = freq

    @property
    def phase(self) -> Quantity:
        """The phase of the pplse.

        Returns
        -------
        Quantity
            The phase of the pulse.
        """
        return self.__phase

    @phase.setter
    def phase(self, phase: Quantity) -> None:
        """Set the phase of the DRAG signal.

        Parameters
        ----------
        cthree.Quantity
            The phase value to be set.

        """
        self.__phase = phase

    @staticmethod
    def __add_deltas(
        envelopeTones: list[Waveform], deltas: list[Quantity]
    ) -> list[Waveform]:
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
                    min_value=np.array(-10.0 * 200e6 * 2 * np.pi),
                    max_value=np.array(-0.1 * 200e6 * 2 * np.pi),
                    unit="Hz",
                    name="Delta",
                ),
            )
        return envelopeTones

    @staticmethod
    def __getToneDelta(tone: Waveform) -> Quantity:
        """Return a list of deltas for each tone.

        Returns
        -------
        Quantity
            List of delta values for each tone.
        """
        return tone.__getattribute__("_" + tone.__class__.__name__ + "__delta")

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
        for tone in self.__envelopeTones:
            params.extend(tone.getParameters())
        params.append(self.__phase)
        for tone in self.__envelopeTones:
            params.append(self.__getToneDelta(tone))
        params += self.__carrierTone.getParameters()
        return params

    def __getOwnParameters(self) -> list[Quantity]:
        """Return a list of DRAGGenerator parameters.

        Returns the phase and deltas of the tones.

        Returns
        -------
        list[Quantity]
            Parameters of DRAGGenerator.
        """
        params = list()
        params.append(self.__phase)
        for tone in self.__envelopeTones:
            params.append(self.__getToneDelta(tone))
        return params

    def setOptimisableParameters(self, params: list[Quantity]) -> None:
        """Set specified parameters to be optimised.

        Also add the indices to `__gradArgNums` to compute the gradients.

        Parameters
        ----------
        params : list[Quantity]
        """
        super().setOptimisableParameters(params)

        self.__carrierTone.setOptimisableParameters(params)

        for tone in self.__envelopeTones:
            tone.setOptimisableParameters(params)

        # Recompute the gradient function
        self.__computeGradientFunc()

    def __computeSignal(self, t, phase, *deltas):
        """Compute signal with DRAG based envelope function and phase.

        Explicit function of parameters to compute gradients using AD.

        Parameters
        ----------
        t : np.npdarray
            time array
        phase : float
            phase of the signal
        deltas: List[float]
            Variable number of inputs for delta parameters for each tone.

        Returns
        -------
        np.ndarray
            Signal as a function of time
        """
        IQ_signal = self._dragEnvelope(t, *deltas) * jnp.exp(1j * phase)
        carr_signal = self.__carrierTone.computeOutput(t)
        # carr_inphase * inphase + carr_quadrature * quadrature
        return jnp.squeeze(jnp.real(IQ_signal.conj() * carr_signal))

    def generateSignal(self, t: np.ndarray) -> np.ndarray:
        """Compute a DRAG corrected signal in the shape of the EnvelopeTone.

        Parameters
        ----------
        t: np.ndarray
            One-dimensional vector of timestamps.

        Returns
        -------
        np.ndarray
            Returns a vector signal of the DRAG corrected signal.
        """
        phase = self.__phase.getValue()
        deltas = [
            self.__getToneDelta(tone).getValue()
            for tone in self.__envelopeTones
        ]
        return self.__computeSignal(t, phase, *deltas)

    def __computeGradientFunc(self):
        """Compute Gradient function for AD."""
        self.__gradArgNums = ()

        ownParams = self.__getOwnParameters()
        for i, param in enumerate(ownParams):
            if self._isOptimised(param):
                self.__gradArgNums += (i + 1,)

        if len(self.__gradArgNums) > 0:
            self.__gradientFunction = grad(
                self.__computeSignal, argnums=self.__gradArgNums
            )

    def __getParameterScale(self) -> list[np.ndarray]:
        """Return parameter scale of optimised parameters.

        Returns parameter scales of phases and deltas if they are optimised.

        Returns
        -------
        list[numpy.ndarray]
            Parameter scales of optimised parameters.
        """
        param_scales = []
        ownParams = self.__getOwnParameters()
        for param in ownParams:
            if self._isOptimised(param):
                param_scales.append(param.getScale())
        return param_scales

    def generateSignalGradient(self, t: np.ndarray) -> jnp.ndarray:
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
        phase = self.__phase
        deltas = [self.__getToneDelta(tone) for tone in self.__envelopeTones]
        deltas_values = [delta.getValue() for delta in deltas]

        grads = []
        gradients = jnp.zeros(shape=(t.shape[0], 0))
        for dev in self.__envelopeTones:
            grads = dev.computeGradient(t)
            gradients = jnp.append(gradients, grads, axis=1)

        grad_carrier = self.__carrierTone.computeGradient(t)
        gradients = jnp.append(gradients, grad_carrier, axis=1)

        if self.__gradientFunction is not None:
            parameter_scales = jnp.array(self.__getParameterScale())
            parameter_scales = jnp.reshape(
                parameter_scales, (1,) + parameter_scales.shape
            )
            grads = vmap(
                self.__gradientFunction,
                in_axes=(0, None) + (None,) * len(deltas),
            )(t, phase.getValue(), *deltas_values)
            grads = jnp.stack(grads, axis=1)
            grads = jnp.squeeze(grads) * parameter_scales

        gradients = jnp.append(gradients, grads, axis=1)
        return jnp.array(gradients)

    def generateSignalGradientOneTime(self, t: np.ndarray) -> jnp.ndarray:
        """Generate gradient of the signal for one time point.

        Collect and return the parameter gradients from the Tone and the carrier
        Tone. Compute the gradient of the generator parameters by AD.
        The order of the gradients should match the order of paramters in
        `self.getParameter()` method

        Parameters
        ----------
        t : float
            A single time point.

        Returns
        -------
        jnp.ndarray
            Array of gradients wrt each parameter for each time point.
        """
        phase = self.__phase
        deltas = [self.__getToneDelta(tone) for tone in self.__envelopeTones]
        deltas_values = [delta.getValue() for delta in deltas]

        # Envelope gradients
        gradients = jnp.zeros(shape=(0,))
        for dev in self.__envelopeTones:
            grads = jnp.squeeze(dev.computeGradient(t), axis=0)
            gradients = jnp.append(gradients, grads, axis=0)

        # Gradient of phase and deltas
        if self.__gradientFunction is not None:
            parameter_scales = jnp.array(self.__getParameterScale())

            grads = jnp.stack(
                self.__gradientFunction(t, phase.getValue(), *deltas_values),
                axis=0,
            )
            grads = jnp.squeeze(grads) * parameter_scales
            gradients = jnp.append(gradients, grads, axis=0)

        # Gradient of carrier signal
        grad_carrier = jnp.squeeze(
            self.__carrierTone.computeGradient(t), axis=0
        )
        gradients = jnp.append(gradients, grad_carrier, axis=0)

        return jnp.array(gradients)
