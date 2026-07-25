import os
import sys
from time import time
import numpy as np
import torch as pt
from utils.model import MyModel, get_n_params
from utils.ll_wrapper import LLWrapper
from utils.cosyne_wrapper import CosyneWrapper
from utils.logging import Context, Logger, HDF5Backend, SustainedCondition
from utils.optimization import init_directories, write_pop_evals, write_map_evals, write_params, write_final_params, write_replaced_params, copy_source_files, create_fields, evaluate
from evotorch_custom.algorithms import MAPElites

       
if __name__ == "__main__":
    test_run = not sys.argv[1] == 'False' if len(sys.argv) > 1 else True
    _, results_path = init_directories("./data/exp/", test_run=test_run, min_id=0)
    n_agents = 60
    n_evals = 16
    max_frames = 1000
    n_epochs = 5000

    if test_run:
        n_agents = 6
        max_frames = 100
        n_evals = 8
        n_epochs = 10

    batch_size = n_agents * n_evals
    device = "cuda" if pt.cuda.is_available() else "cpu"
    n_threads = max([i for i in range(1, max(4, os.cpu_count() // 4) + 1) if batch_size % i == 0])

    som_path = "./data/soms/small/random_play_sigma_4_uniform_1k_small_max.npz"
    som_data = np.load(som_path, allow_pickle=True)
    weights = pt.from_numpy(som_data['weights']).to(device)
    means = pt.from_numpy(som_data['input_means']).to(device)
    stds = pt.from_numpy(som_data['input_stds']).to(device)

    state_dict = {
        "som.input_means": means,
        "som.input_stds": stds,
        "som.weights": weights,
        "chip.Win_scale": pt.full((batch_size,), -2.5),
        "chip.Win_bias": pt.full((batch_size,), 5),
    }

    model = MyModel(batch_size, device, 8, 4, 20, reservoir_n_in=weights.shape[0], reservoir_n=128)
    model.load_state_dict(state_dict, strict=False)

    env = LLWrapper(batch_size, n_threads, device)

    n_params = get_n_params(model)
    map_lower = [0.25, 0.01]
    map_upper = [0.95, 0.25]
    n_map_bins = 17
    map_depth = 2
    feature_grid = MAPElites.make_feature_grid(map_lower, map_upper, n_map_bins, dtype = 'float32', device=device)

    cos = CosyneWrapper(feature_grid, map_depth, n_agents, n_params, evals_size=3, device=device)

    if not test_run:
        source_files = ["optimize.py", "./utils/*.py"]
        copy_source_files(source_files, os.path.join(results_path, "sources"))
    
    buffers = {k: v for k, v in model.state_dict().items() if k in dict(model.named_buffers())}
    pt.save(buffers, os.path.join(results_path, 'model.pt'))

    # Init logger
    saved_params = [k.replace("parametrizations.", "").replace(".original", "") for k, v in model.named_parameters() if v.numel() < 10 * batch_size]
    ctx = Context(model=model, env=env, cos=cos, n_epochs=n_epochs, saved_params=saved_params)

    backend = HDF5Backend(os.path.join(results_path, "results.h5"), compression="gzip", compression_opts=4, dtype="float32")
    create_fields(backend, ctx)

    logger = Logger(backend)
    logger.add("after_eval", write_pop_evals)
    logger.add("after_evolution", write_map_evals)
    logger.add("after_evolution", write_params)
    logger.add("after_evolution", write_final_params, trigger=lambda ctx: ctx.ep % max(5, (ctx.n_epochs // 100)) == 0 or ctx.ep == ctx.n_epochs)
    logger.add("after_evolution", write_replaced_params, trigger=SustainedCondition(lambda ctx: ctx.pop_best_eval > -20, 3000))

    # Start optimization
    params = cos.initial_params()
    for ep in range(n_epochs):
        t_start = time()
        ctx.ep = ep

        evals = evaluate(model, env, params, n_evals=n_evals, max_frames=max_frames, device=device)
        cos.set_evals(pt.stack(evals, dim=1))
        logger.emit("after_eval", ctx)
        
        ctx.mean_eval = cos.searcher.status["mean_eval"]
        ctx.pop_best_eval = cos.searcher.status["pop_best_eval"]
        print(f"ep{ep}:", np.round(time() - t_start, 2), "sec", end=';   ')
        print("mean:", round(ctx.mean_eval, 2), "best:", round(ctx.pop_best_eval, 2))


        params = cos.step(evals)
        logger.emit("after_evolution", ctx)
