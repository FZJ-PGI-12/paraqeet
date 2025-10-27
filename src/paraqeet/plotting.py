"""A plotting library for ParaQeet."""

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import matplotlib as mpl

import numpy as np
import json

import jax.numpy as jnp
from jax import vmap

from paraqeet.propagation.propagation import Propagation
from paraqeet.quantity import Array
from paraqeet.signal.waveform import Waveform
from paraqeet.signal.generator import Generator

matplotlib_inline.backend_inline.set_matplotlib_formats("pdf", "svg")

# Specifying the custom plotting fonts
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = "Tex Gyre Pagella"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.size"] = " 10.0"
mpl.rcParams["axes.labelsize"] = " 11.0"
mpl.rcParams["legend.fontsize"] = "10.0"
mpl.rcParams["xtick.labelsize"] = "10.0"
mpl.rcParams["ytick.labelsize"] = "10.0"
mpl.rcParams["axes.linewidth"] = "1.0"
mpl.rcParams["xtick.major.size"] = "3.0"
mpl.rcParams["xtick.major.width"] = "1.0"
mpl.rcParams["ytick.major.size"] = "3.0"
mpl.rcParams["ytick.major.width"] = "1.0"
mpl.rcParams["axes.titlesize"] = "medium"
mpl.rcParams["figure.titlesize"] = "medium"

# Specifying a custom color palette
default_colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
custom_colors = [
    "#18428A",
    "#FF9258",
    "#57B870",
    "#BF433B",
    "#FF00A6",
    "#FFCC33",
    "#008B8B",
    "#9B5DE5",
] + default_colors

mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=custom_colors)  # type: ignore


def plot_signal_and_dynamics(
    generator: Generator,
    propagation: Propagation,
    times: Array,
    state_labels: list[str] | None = None,
    axes=None,
    linestyle: str = "-",
    label: str = "",
    alpha: float = 1.0,
    linewidth: float = 1.5,
):
    """Plot the signal and the correspoding dynamics.

    If fig or ax is provided then ax[0] is used to plot the signal and ax[1] for dynamics.
    This can be used to plot multiple signals and dyanmics on the same plot.
    """

    def calculate_populations(states, dm=False):
        """Calculate state populations from density matrices and vectorized dm."""
        if len(states.shape) > 2:
            if dm:
                pops = jnp.abs(vmap(jnp.diag, in_axes=0)(states))
            else:
                pops = jnp.abs(states[:, :, 0]) ** 2
                pops = jnp.reshape(pops, [pops.shape[0], pops.shape[1]])
        else:
            if dm:
                pops = jnp.diag(states)
            else:
                pops = jnp.abs(states) ** 2
        return pops

    states = propagation.propagate(times)
    sig = generator.generate_signal(times) / 1e6 / (2 * np.pi)

    if axes is None:
        _, axes = plt.subplots(2, figsize=(4, 5), sharex=True, dpi=100)

    axes[0].plot(
        times / 1e-9,
        np.real(sig),
        label="Re " + label,
        ls=linestyle,
        alpha=alpha,
        linewidth=linewidth,
    )
    axes[0].plot(
        times / 1e-9,
        np.imag(sig),
        label="Imag " + label,
        ls=linestyle,
        alpha=alpha,
        linewidth=linewidth,
    )
    axes[0].legend(loc=1)
    axes[0].set_ylabel("Amplitude \n" + r"[MHz / $2\pi$]")
    axes[0].grid(True, linestyle=(1, (1, 5)), linewidth=1)

    axes[1].plot(
        times / 1e-9,
        calculate_populations(states, dm=propagation.is_open),
        ls=linestyle,
        alpha=alpha,
        linewidth=linewidth,
    )
    axes[1].set_ylabel("Population")
    axes[-1].set_xlabel("Time [ns]")
    if state_labels is not None:
        axes[1].legend(state_labels)
    axes[1].grid(True, linestyle=(1, (1, 5)), linewidth=1)

    return axes


def plot_signal(
    device: Generator | Waveform,
    times: Array,
    axes=None,
    linestyle: str = "-",
    label: str = "",
    alpha: float = 1.0,
    linewidth: float = 1.5,
):
    """Plot signal from Generator or Envelope."""
    if isinstance(device, Generator):
        sig = device.generate_signal(times) / 1e6 / (2 * np.pi)
    else:
        sig = device.compute_output(times) / 1e6 / (2 * np.pi)

    if axes is None:
        _, axes = plt.subplots(1, figsize=(4, 3), dpi=100)

    axes.plot(
        times / 1e-9,
        np.real(sig),
        label="Re " + label,
        ls=linestyle,
        alpha=alpha,
        linewidth=linewidth,
    )
    axes.plot(
        times / 1e-9,
        np.imag(sig),
        label="Imag " + label,
        ls=linestyle,
        alpha=alpha,
        linewidth=linewidth,
    )
    axes.grid(True, linestyle=(1, (1, 5)), linewidth=1)
    axes.set_ylabel(r"Amplitude [MHz / $2\pi$]")
    axes.set_xlabel("Time [ns]")
    axes.legend()
    return axes


def plot_infidelity_vs_evaluation_from_logs(
    log_path: str,
    axes=None,
    linestyle: str = "-",
    label: str = "",
    alpha: float = 1.0,
    linewidth: float = 1.5,
):
    """Plot Infidelity vs Evaluation from json log file.

    Parameters
    ----------
    log_path : str
        Path to json log file.
    """
    with open(log_path) as file:
        log = [json.loads(line) for line in file]

    infidelities = []
    for eval in log:
        infidelities.append(eval["Goal"])

    if axes is None:
        _, axes = plt.subplots(1, figsize=(4, 4), sharex=True, dpi=100)

    axes.plot(
        range(len(infidelities)),
        infidelities,
        label=label,
        ls=linestyle,
        alpha=alpha,
        linewidth=linewidth,
    )
    axes.axhline(np.min(infidelities), color="red", ls="--", alpha=0.7)
    axes.set_yscale("log")
    axes.set_ylim(bottom=0.1 * min(infidelities))
    axes.text(x=len(infidelities) * 0.1, y=0.2 * min(infidelities), s=f"Minimum infidelity = {min(infidelities):.3e}")

    axes.legend(loc=1)
    axes.set_ylabel("Infidelity")
    axes.set_xlabel("Evaluation number")
    axes.grid(True, linestyle=(1, (1, 5)), linewidth=1)
    return axes
