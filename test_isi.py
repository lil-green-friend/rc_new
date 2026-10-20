import os
from time import time

import h5py
import torch as pt

from optimize import evaluate
from utils.ll_wrapper import LLWrapper
from utils.logging import Context, HDF5Backend, Logger
from utils.to_model.utils import instanciate_model

if __name__ == "__main__":
    model_path = "data/exp/22/model.pt"
    results_path = "data/exp/22/results.h5"

    input_spike_distances = [i for i in range(20)]

    batch_size = 1

    device = "cuda" if pt.cuda.is_available() else "cpu"
    n_evals = 1
    max_frames = 4000

    n_threads = max(
        [i for i in range(1, max(4, os.cpu_count() // 4) + 1) if batch_size % i == 0]
    )

    f = h5py.File(results_path)
    model = instanciate_model(
        model_path,
        input_spike_interval=input_spike_distances[0],
        batch_size=batch_size,
        device=device,
    )
    env = LLWrapper(batch_size, n_threads, device)

    c1, c2, d = 10, 10, 0

    original_evals = pt.tensor(
        [
            f[path][-1, c1, c2, d].item()
            for path in [
                "map_evals/fitness",
                "map_evals/total_score",
                "map_evals/action_prob",
                "map_evals/firing_rate",
            ]
        ]
    )

    params = pt.from_numpy(f["final_params"][c1, c2, d])[None].to(pt.float)

    ctx = Context()

    backend = HDF5Backend(
        "data/spike_plots/tiny_odin.h5",
        compression="gzip",
        compression_opts=4,
        dtype="float32",
    )
    # backend.add_path("spikes", shape=(len(input_spike_intervals),max_frames,20,256), chunks=(1,1,20,256))
    backend.add_path("evals", shape=(len(input_spike_distances), 4), chunks=(1, 4))
    backend.add_path("input_spike_intervals", shape=len(input_spike_distances))
    backend.add_path("original_evals", shape=(4,))
    backend.write("original_evals", original_evals)

    def save_spikes(backend, ctx):
        backend.write_idx(
            "spikes", (ctx.isi_idx, ctx.spike_idx), ctx.eval_locals["activity"][0].T
        )
        ctx.spike_idx += 0

    def save_evals(backend, ctx):
        backend.write_next("evals", ctx.evals)
        backend.write_next("input_spike_intervals", ctx.isi)

    logger = Logger(backend)
    # logger.add("eval__after_forward", save_spikes)
    logger.add("eval_finished", save_evals)

    for isi_idx, isd in enumerate(input_spike_distances):
        input_spike_interval = (1 + 512) * 2 * isd
        t_start = time()
        ctx.isi = isd
        ctx.isi_idx = isi_idx
        ctx.spike_idx = 0
        sd = {"input_spike_interval": pt.tensor(input_spike_interval)}
        model.load_state_dict(sd, strict=False)
        evals = evaluate(
            model,
            env,
            params,
            n_evals=n_evals,
            max_frames=max_frames,
            logger=logger,
            ctx=ctx,
            device=device,
        )
        ctx.evals = pt.tensor(evals)
        print(model.input_spike_interval)
        logger.emit("eval_finished", ctx)
        print("isi:", isd, "evals:", ctx.evals, "time:", time() - t_start)

    print("Original evals:", original_evals)
