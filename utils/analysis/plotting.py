import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from numpy.typing import NDArray


def savefig(path, **kw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, **kw)


def spike_scatterplot(  # TODO test
    ax: Axes,
    data: NDArray[np.number],
    t_start: int = 0,
    neuron_colors: Any | None = None,
    param_dict: dict[str, Any] | None = None,
) -> PathCollection:
    """Plot nonzero entries in a spike-activity array.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes on which to draw the scatter plot.
    data : NDArray[np.number]
        Array of shape ``(n_sub_steps, n_neurons)``. Nonzero entries are
        plotted as spikes.
    neuron_colors : Any, optional
        A single color or one color for each neuron. Cannot be combined with
        the ``"c"`` parameter in ``param_dict``.
    param_dict : dict[str, Any], optional
        Additional keyword arguments passed to ``ax.scatter``.

    Returns
    -------
    PathCollection
        The scatter plot artist returned by Matplotlib.
    """

    if param_dict is None:
        param_dict = {}
    else:
        param_dict = param_dict.copy()

    spike_steps, neurons = data.nonzero().T

    if neuron_colors is not None:
        assert "c" not in param_dict, (
            "Expecting either neuron_colors or c as a parameter"
        )
        color_values = np.asarray(neuron_colors)
        if (
            color_values.ndim > 0 and color_values.size != 4
        ):  # TODO this probably does not cover all edge cases
            assert color_values.size == data.shape[1], (
                f"Expecting either color for each neuron or a single color for all neurons, got: {color_values}"
            )
            neuron_colors = color_values[neurons]

        param_dict["c"] = neuron_colors

    out = ax.scatter(spike_steps + t_start, neurons, **param_dict)
    ax.set_xlabel("n_sub_steps")

    return out
