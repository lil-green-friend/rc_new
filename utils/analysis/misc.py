import importlib
import inspect
from collections.abc import Callable

import h5py
import numpy as np
import torch as pt

from utils.model import set_params


def randomly_select_final_agents_by_fitness(
    run, minimum_fitness, n_agents=None, allow_duplicates=False
):
    assert n_agents is None or n_agents > 0, "Zero agents not allowed"

    f = h5py.File(f"data/exp/{run}/results.h5")

    fitness = f["map_evals/fitness"][-1]

    valid_indices = np.argwhere(fitness > minimum_fitness)

    n_to_select = 1 if n_agents is None else n_agents

    if (not allow_duplicates and len(valid_indices) < n_to_select) or (
        allow_duplicates and len(valid_indices) == 0
    ):
        raise ValueError(
            f"Found only {len(valid_indices)} agents with fitness > "
            f"{minimum_fitness}, but {n_to_select} were requested"
        )

    selected_indices = np.random.choice(
        len(valid_indices), size=n_to_select, replace=allow_duplicates
    )
    if n_agents is None:
        return valid_indices[selected_indices[0]]
    return np.transpose(valid_indices[selected_indices])  # can be used as arr[*idx]


def get_model_from_final_params(run):
    with h5py.File(f"data/exp/{run}/results.h5") as f:
        params = f["final_params"]
    return get_model(run, pt.from_numpy(params).float().flatten(end_dim=-2))


def get_model_from_indices(run, idx):
    with h5py.File(f"data/exp/{run}/results.h5") as f:
        idx = np.asarray(idx)
        final_params = f["final_params"]
        if idx.ndim == 1:
            params = final_params[tuple(idx.tolist())]
        else:
            # h5py requires sorted indices and does not support duplicate indices.
            params = np.stack([final_params[tuple(coord.tolist())] for coord in idx.T])
    return get_model(run, pt.from_numpy(params).float())


def get_model(run_id, params=None):
    base_path = f"data/exp/{run_id}"
    if params is not None:
        if type(params) == np.ndarray:
            params = pt.from_numpy(params)
        if len(params.shape) == 1:
            params = params.unsqueeze(0)
        bs = params.shape[0]
        m, _ = instanciate_model(base_path, bs)
        set_params(m, params, bs)
    else:
        m, _ = instanciate_model(base_path, 1)
    return m


def instanciate_model(
    source_path, batch_size, device="cpu"
) -> tuple[pt.nn.Module, Callable]:
    sources = source_path + "/sources/utils/model"
    optim = source_path + "/sources/optimize"

    model_module = importlib.import_module(sources.replace("/", "."))
    optim_module = importlib.import_module(optim.replace("/", "."))

    MyModel = model_module.MyModel
    eval = optim_module.evaluate

    state_dict = pt.load(source_path + "/model.pt", map_location=pt.device(device))

    if "37" in source_path:
        state_dict["chip.parametrizations.Win_scale.original"] = state_dict[
            "chip.parametrizations.Win_scale.original"
        ][0:1]
        state_dict["chip.parametrizations.Win_bias.original"] = state_dict[
            "chip.parametrizations.Win_bias.original"
        ][0:1]

    # Filter sd to only include keys that match MyModel's __init__ parameters
    model_init_params = inspect.signature(MyModel.__init__).parameters
    filtered_sd = {k: v for k, v in state_dict.items() if k in model_init_params}

    m = MyModel(batch_size, "cpu", **filtered_sd)
    m.load_state_dict(state_dict, strict=False)
    return m, eval
