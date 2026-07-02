import os
import shutil
import numpy as np
import torch as pt
import torch.nn.utils.parametrize as parametrize
from .model import get_n_params, get_param_offset_and_size, set_params
import glob


def evaluate(model, env, params, n_evals, max_frames, device="cpu", logger=None, ctx=None):
    w = pt.linspace(1 / n_evals, 1, n_evals)
    w = w / pt.sum(w)
    # TODO with torch.inference_mode() ???
    
    params_stacked = pt.repeat_interleave(params, n_evals, dim=0)

    bs = params_stacked.shape[0]
    set_params(model, params_stacked, bs)
    # model.reset()

    rewards = pt.zeros((max_frames, bs), dtype=pt.float, device=device)
    action_prob = pt.zeros((max_frames, bs), dtype=pt.bool, device=device)
    spike_activity = pt.zeros((max_frames, bs), dtype=pt.float, device=device)

    last_term = pt.zeros(bs, dtype=pt.int, device=device) - 1
    term_count = pt.zeros(bs, dtype=pt.int, device=device)
    
    if logger is not None:
        ctx.eval_locals = locals()
        logger.emit("eval__before_eval", ctx)

    obs = env.reset()
    with parametrize.cached():
        for f in range(max_frames):
            action, activity = model(obs)
            if logger is not None:
                ctx.eval_locals = locals()
                logger.emit("eval__after_forward", ctx)
            obs, rew, term, _ = env.step(action)
            rewards[f] = rew
            action_prob[f] = action > 0

            spike_activity[f] = pt.mean(activity, [1, 2])
            last_term[term] = f 
            term_count += term 
       
    total_score = pt.sum(rewards, 0)
    for i in range(bs):
        # if at least one episode was completed, cut off rewards for last, uncompleted episode
        if last_term[i] != -1:
            rewards[last_term[i]+1:, i] = 0
    per_episode_score = pt.sum(rewards, 0) / pt.clip(pt.tensor(term_count), min=1)
    action_prob = pt.mean(action_prob.float(), 0)
    firing_rate = pt.mean(spike_activity.float(), 0)

    per_episode_score, total_score, action_prob, firing_rate = [x.reshape(bs // n_evals, n_evals) for x in [per_episode_score, total_score, action_prob, firing_rate]]

    fitness = fitness_function(per_episode_score, w)

    total_score, action_prob, firing_rate = [pt.mean(x, dim=1) for x in [total_score, action_prob, firing_rate]]

    return fitness, total_score, action_prob, firing_rate

def fitness_function(norm_score, w):
    """
    Calculates fitness as weighted average to counteract games where agents got lucky
    """
    sorted_scores, _ = pt.sort(norm_score, dim=1, descending=True)
    weighted_fitness = pt.sum(sorted_scores * w, 1)
    return weighted_fitness

def create_fields(backend, ctx):
    n_epochs = ctx.n_epochs
    map = ctx.cos.searcher.map
    map_shape = (map._bin_size, map._bin_size, map._depth)

    for key in ("pop_best_eval", "median_eval", "mean_eval"):
        backend.add_path(f"results/{key}", shape=(n_epochs,), chunks=(n_epochs,))

    for key in ("fitness", "total_score", "action_prob", "firing_rate"):
        backend.add_path(f"map_evals/{key}", shape=(n_epochs,) + map_shape, chunks=(1,) + map_shape)

    o_and_s = get_param_offset_and_size(ctx.model)
    for key in ctx.saved_params:
        per_epoch_shape = map_shape + (o_and_s[key][1],)
        backend.add_path(f"params/{key}", shape=(n_epochs,) + per_epoch_shape, chunks=(1,) + per_epoch_shape,)

    s = get_n_params(ctx.model)
    backend.add_path("final_params", shape=map_shape + (s,), chunks=(1, 1, 1, s), dtype="float64")

    for key in ["ep", "depth", "cell"]:
        backend.add_path(f"replaced_params/{key}", dtype="float64")
    backend.add_path("replaced_params/params", shape=(0, s), chunks=(1, s), dtype="float64")

def write_pop_evals(backend, ctx):
    for key in ("pop_best_eval", "median_eval", "mean_eval"):
        backend.write_next(f"results/{key}", ctx.cos.searcher.status[key])

def write_map_evals(backend, ctx):
    evals = ctx.cos.get_map_evals()
    for i, k in enumerate(("fitness", "total_score", "action_prob", "firing_rate")):
        backend.write_next(f"map_evals/{k}", evals[..., i])

def write_final_params(backend, ctx):
    backend.write("final_params", ctx.cos.get_map_solutions())

def write_params(backend, ctx):
    o_and_s = get_param_offset_and_size(ctx.model)
    params = ctx.cos.get_map_solutions()
    for name in ctx.saved_params:
        offset, size = o_and_s[name]
        backend.write_next(f"params/{name}", params[..., offset : offset + size])

def write_replaced_params(backend, ctx):
    replaced = pt.nonzero(ctx.cos.searcher.map._age == 1)
    for depth, idp in replaced:
        params = ctx.cos.searcher.map._map[depth][idp].values
        backend.write_next("replaced_params/ep", ctx.ep)
        backend.write_next("replaced_params/depth", depth)
        backend.write_next("replaced_params/cell", idp)
        backend.write_next("replaced_params/params", params)
    
def vectorize(agents, device):
    params_vec = {}
    for p in agents[0].keys():
        ps = [a[p] for a in agents]
        p_vec = np.stack(ps) if isinstance(ps[0], np.ndarray) else np.array(ps)
        params_vec[p] = pt.from_numpy(p_vec).to(device)
    return params_vec

def init_directories(base_dir: str, test_run: bool, min_id=0) -> tuple[int | str, str]:
    """Initializes a directory for saving results. The directory is named after the run_id. 
    The run_id which is auto-incremented, unless "test_run=True is passed. 
    In this case, the directory is named 'test' and the contents are deleted if they already exist.
    This can be used to test if the code works without creating a new directory each time.
    """
    if test_run:
        run_id = 'test'     
    else:
        # auto increment run id
        existing_run_ids = [int(name) for name in os.listdir(base_dir) if name.isdigit() and os.path.isdir(os.path.join(base_dir, name))]
        run_id = max(existing_run_ids) + 1 if existing_run_ids else 0
        run_id = max(run_id, min_id)
        
    save_dir = os.path.join(base_dir, str(run_id))

    print(f"Saving to {save_dir}")
    
    if test_run and os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    return run_id, save_dir

def copy_source_files(source_files, target_path):
    for src in source_files:
        for file in glob.glob(src):
            rel_path = os.path.relpath(file, start=os.curdir) 
            new_path = os.path.dirname(os.path.join(target_path, rel_path))
            os.makedirs(new_path, exist_ok=True)
            shutil.copy(file, new_path)