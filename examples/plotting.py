"""A plotting library for ParaQeet."""

import json

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import numpy as np
from jax import vmap

from paraqeet.propagation.propagation import Propagation
from paraqeet.propagation.utils import convert_vec_to_dm
from paraqeet.quantity import Array
from paraqeet.signal.generator import Generator
from paraqeet.signal.signal import Signal

matplotlib_inline.backend_inline.set_matplotlib_formats("png")

# Specifying the custom plotting fonts
mpl.rcParams["font.family"] = "serif"
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
plt.rcParams["figure.dpi"] = 125  # Set default DPI for all figures
plt.rcParams["savefig.dpi"] = 300  # Higher DPI for saved files

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
    marker: str = "",
    open_system: bool = False,
    vectorized_dm: bool = False,
    tol: float | None = None,
):
    """Plot the signal and the corresponding dynamics.

    If fig or ax is provided then ax[0] is used to plot the signal and ax[1] for dynamics.
    This can be used to plot multiple signals and dynamics on the same plot.
    """

    def calculate_populations(states, open_system, vectorized_dm):
        """Calculate state populations from density matrices and vectorized dm."""
        if vectorized_dm:
            states = vmap(convert_vec_to_dm, in_axes=(0))(states)
        if len(states.shape) > 2:
            if open_system:
                pops = jnp.abs(vmap(jnp.diag, in_axes=0)(states))
            else:
                pops = jnp.abs(states[:, :, 0]) ** 2
                pops = jnp.reshape(pops, [pops.shape[0], pops.shape[1]])
        else:
            if open_system:
                pops = jnp.diag(states)
            else:
                pops = jnp.abs(states) ** 2
        return pops

    states = propagation.propagate(times)
    sig = generator.get_value(times) / 1e6 / (2 * np.pi)

    if axes is None:
        _, axes = plt.subplots(2, figsize=(4, 5), sharex=True)

    axes[0].plot(
        times / 1e-9, np.real(sig), label="Re " + label, ls=linestyle, alpha=alpha, linewidth=linewidth, marker=marker
    )
    axes[0].plot(
        times / 1e-9, np.imag(sig), label="Imag " + label, ls=linestyle, alpha=alpha, linewidth=linewidth, marker=marker
    )
    axes[0].legend(loc=1)
    axes[0].set_ylabel("Amplitude \n" + r"[MHz / $2\pi$]")
    axes[0].grid(True, linestyle=(1, (1, 5)), linewidth=1)

    pops = calculate_populations(states, open_system, vectorized_dm)
    if tol is not None:
        max_pops = np.max(pops, axis=0)
        index_mask = [True if pop > tol else False for pop in max_pops]
    else:
        index_mask = [True] * pops.shape[1]

    axes[1].plot(
        times / 1e-9,
        pops[:, np.array(index_mask)],
        ls=linestyle,
        alpha=alpha,
        linewidth=linewidth,
        marker=marker,
    )
    axes[1].set_ylabel("Population")
    axes[-1].set_xlabel("Time [ns]")
    if state_labels is not None:
        axes[1].legend([state_label for i, state_label in enumerate(state_labels) if index_mask[i]])
    axes[1].grid(True, linestyle=(1, (1, 5)), linewidth=1)

    return axes


def plot_signal(
    device: Generator | Signal,
    times: Array,
    axes=None,
    linestyle: str = "-",
    label: str = "",
    alpha: float = 1.0,
    linewidth: float = 1.5,
    marker: str = "",
):
    """Plot signal from Generator or Envelope."""
    if isinstance(device, Generator):
        sig = device.get_value(times) / 1e6 / (2 * np.pi)
    else:
        sig = device.get_value(times) / 1e6 / (2 * np.pi)

    if axes is None:
        _, axes = plt.subplots(1, figsize=(4, 3))

    axes.plot(
        times / 1e-9,
        np.real(sig),
        label="Re " + label,
        ls=linestyle,
        alpha=alpha,
        linewidth=linewidth,
        marker=marker,
    )
    axes.plot(
        times / 1e-9,
        np.imag(sig),
        label="Imag " + label,
        ls=linestyle,
        alpha=alpha,
        linewidth=linewidth,
        marker=marker,
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
    marker: str = "",
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
        _, axes = plt.subplots(1, figsize=(4, 4), sharex=True)

    axes.plot(
        range(len(infidelities)),
        infidelities,
        label=label,
        ls=linestyle,
        alpha=alpha,
        linewidth=linewidth,
        marker=marker,
    )
    axes.axhline(np.min(infidelities), color="red", ls="--", alpha=0.7)
    axes.set_yscale("log")
    axes.set_ylim(bottom=0.7 * min(infidelities))
    axes.text(x=len(infidelities) * 0.1, y=0.8 * min(infidelities), s=f"Minimum infidelity = {min(infidelities):.3e}")

    axes.legend(loc=1)
    axes.set_ylabel("Infidelity")
    axes.set_xlabel("Evaluation number")
    axes.grid(True, linestyle=(1, (1, 5)), linewidth=1)
    return axes
