import os
import shutil
import sys
from time import time
import h5py
import numpy as np
import torch as pt
import torch.nn.utils.parametrize as parametrize
from utils.model import MyModel, get_n_params, get_param_offset_and_size
from utils.ll_wrapper import LLWrapper
from utils.cosyne_wrapper import CosyneWrapper
from evotorch.algorithms import MAPElites
import glob

def set_params(model, vector, bs):
    pointer = 0
    for p in model.parameters():
        num = p.numel() // bs
        p.data = vector[:, pointer:pointer+num].view_as(p)
        pointer += num

    model.update_parametrizations()

def evaluate(model, env, params, n_evals, max_frames, device="cpu"):
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
    
    obs = env.reset()
    with parametrize.cached():
        for f in range(max_frames):
            action, activity = model.forward(obs)            
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

def create_fields(f: h5py.File, n_epochs: int, model: pt.nn.Module, map, saved_params: list):
    res = f.create_group("results")
    for key in ("pop_best_eval", "median_eval", "mean_eval"):
        res.create_dataset(key, shape=(n_epochs,), dtype="float32",
                           chunks=(n_epochs,))
        
    map_shape = (map._bin_size, map._bin_size, map._depth)
    ev = f.create_group("map_evals")
    for key in ("fitness", "total_score", "action_prob", "firing_rate"):
        ev.create_dataset(key, shape=(n_epochs,) + map_shape, dtype="float32",
                          chunks=(1,) + map_shape,
                          compression="gzip", compression_opts=4)

    o_and_s = get_param_offset_and_size(model)
    pm = f.create_group("params")
    for name in saved_params:
        per_epoch_shape = map_shape + (o_and_s[name][1],)
        pm.create_dataset(name,
                            shape=(n_epochs,) + per_epoch_shape,
                            dtype="float32",
                            chunks=(1,) + per_epoch_shape,
                            compression="gzip", compression_opts=4)

    s = get_n_params(model)
    f.create_dataset("final_params",
                        shape=map_shape + (s,),
                        dtype="float32",
                        chunks=(1, 1, 1, s),
                        compression="gzip", compression_opts=4)
    return f


def write_data(f: h5py.File, n_epochs: int, ep: int, cos: CosyneWrapper, model: pt.nn.Module, saved_params: list):

    for key in ("pop_best_eval", "median_eval", "mean_eval"):
        f["results"][key][ep] = cos.searcher.status[key]
    
    evals = cos.get_map_evals()

    f["map_evals"]["fitness"][ep]      = evals[..., 0].cpu().numpy()
    f["map_evals"]["total_score"][ep]  = evals[..., 1].cpu().numpy()
    f["map_evals"]["action_prob"][ep]  = evals[..., 2].cpu().numpy()
    f["map_evals"]["firing_rate"][ep]  = evals[..., 3].cpu().numpy()
    
    o_and_s = get_param_offset_and_size(model)
    params = cos.get_map_solutions()
    for name in saved_params:
        key = name
        offset, size = o_and_s[name]
        f["params"][key][ep] = params[..., offset : offset + size].cpu().numpy()

    if ep % max(5, (n_epochs // 100)) == 0 or ep == n_epochs:
        f["final_params"][...] = params.cpu().numpy()
            

def fitness_function(norm_score, w):
    """
    Calculates fitness as weighted average to counteract games where agents got lucky
    """
    sorted_scores, _ = pt.sort(norm_score, dim=1, descending=True)
    weighted_fitness = pt.sum(sorted_scores * w, 1)
    return weighted_fitness

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
        
if __name__ == "__main__":
    test_run = not sys.argv[1] == 'False' if len(sys.argv) > 1 else True
    _, results_path = init_directories("./data/exp/", test_run=test_run, min_id=0)
    device = "cuda" if pt.cuda.is_available() else "cpu"
    n_agents = 60
    n_threads = max([i for i in range(1, min(4, os.cpu_count() // 2) + 1) if n_agents % i == 0])
    max_frames = 1000
    n_evals = 16
    n_epochs = 3000

    if test_run:
        n_agents = 6
        max_frames = 100
        n_evals = 8
        n_epochs = 10

    som_path = "./data/soms/random_play_sigma_4_uniform_1k.npz"
    som_data = np.load(som_path, allow_pickle=True)
    weights = pt.from_numpy(som_data['weights']).to(device)
    means = pt.from_numpy(som_data['input_means']).to(device)
    stds = pt.from_numpy(som_data['input_stds']).to(device)

    state_dict = {
        "som.input_means": means,
        "som.input_stds": stds,
        "som.weights": weights
    }


    model = MyModel(n_agents*n_evals, device, 8, 4, 20, 32)
    model.load_state_dict(state_dict, strict=False)

    env = LLWrapper(n_agents*n_evals, n_threads, device)

    n_params = get_n_params(model)
    map_lower = [0.25, 0.01]
    map_upper = [0.95, 0.25]
    n_map_bins = 17
    map_depth = 2
    feature_grid = MAPElites.make_feature_grid(map_lower, map_upper, n_map_bins, dtype = 'float32', device=device)

    cos = CosyneWrapper(feature_grid, map_depth, n_agents, n_params, evals_size=3, device=device)

    saved_params = [k.replace("parametrizations.", "").replace(".original", "") for k, v in model.named_parameters() if v.numel() < 10 * n_agents*n_evals]

    params = cos.initial_params()

    source_files = ["optimize.py", "./utils/*.py"]
    if not test_run:
        copy_source_files(source_files, os.path.join(results_path, "sources"))
    

    buffers = {k: v for k, v in model.state_dict().items() if k in dict(model.named_buffers())}
    pt.save(buffers, os.path.join(results_path, 'model.pt'))

    per_epoch = h5py.File(os.path.join(results_path, "results.h5"), "w")
    create_fields(per_epoch, n_epochs + 1, model, cos.searcher.map, saved_params)

    for ep in range(n_epochs + 1):
        t_start = time()
        evals = evaluate(model, env, params, n_evals=n_evals, max_frames=max_frames)
        cos.set_evals(pt.stack(evals, dim=1))

        write_data(per_epoch, n_epochs, ep, cos, model, saved_params)
        
        print(f"ep{ep}:", np.round(time() - t_start, 2), "sec", end=';   ')
        print("mean:", round(cos.searcher.status['mean_eval'], 2), "best:", round(cos.searcher.status['pop_best_eval'], 2))

        params = cos.step(evals)
