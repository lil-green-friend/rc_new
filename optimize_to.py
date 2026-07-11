import os
from time import time
import sys
import h5py
import numpy as np
import torch as pt
from evotorch import Problem
from evotorch.algorithms import Cosyne

from utils.ll_wrapper import LLWrapper
from utils.logging import Context, HDF5Backend, Logger
from utils.optimization import evaluate
from utils.to_model.utils import instanciate_model
from utils.optimization import init_directories
RID = 25
MODEL_PATH = f"data/exp/{RID}/model.pt"
RESULTS_PATH = f"data/exp/{RID}/results.h5"


K = 15
INPUT_SPIKE_INTERVAL = (512 + 1) * 2 * K

POP_SIZE = 30
N_EVALS = 1
MAX_FRAMES = 1000
N_GENERATIONS = 250


def create_fields(backend: HDF5Backend, popsize: int, solution_length: int):
    for key in ("pop_best_eval", "median_eval", "mean_eval"):
        backend.add_path(f"results/{key}", shape=(0,), chunks=(1,))

    backend.add_path(
        "population/params",
        shape=(0, popsize, solution_length),
        chunks=(1, popsize, solution_length),
    )


def write_population(backend: HDF5Backend, ctx: Context):
    population = ctx.searcher.population
    evals = population.access_evals(0)

    backend.write_next("results/pop_best_eval", pt.amax(evals))
    backend.write_next("results/mean_eval", pt.mean(evals))
    backend.write_next("results/median_eval", pt.median(evals))
    backend.write_next("population/params", population.values)


def load_initial_population(results_path: str, pop_size: int) -> pt.Tensor:
    with h5py.File(results_path, "r") as handle:
        all_params = pt.from_numpy(handle["final_params"][:]).to(pt.float32)
        fitness = pt.from_numpy(handle["map_evals/fitness"][:]).to(pt.float32)
        fitness = pt.mean(fitness, dim=-1)

    flat_fitness = fitness.reshape(-1)
    _, flat_indices = pt.topk(flat_fitness, k=pop_size, largest=True, sorted=True)

    width = fitness.shape[1]
    xy_coords = [(int(idx // width), int(idx % width)) for idx in flat_indices]
    z_coords = pt.randint(0, all_params.shape[2], (pop_size,)).tolist()
    selected_coords = [(x, y, int(z)) for (x, y), z in zip(xy_coords, z_coords)]

    selected = pt.stack([all_params[x, y, z] for x, y, z in selected_coords], dim=0)
    return selected


def make_objective(model, env, n_evals: int, max_frames: int, device: str):
    def objective(values: pt.Tensor) -> pt.Tensor:
        fitness, _, _, _ = evaluate(
            model,
            env,
            values,
            n_evals=n_evals,
            max_frames=max_frames,
            device=device,
        )
        return fitness

    return objective


if __name__ == "__main__":
    test_run = not sys.argv[1] == 'False' if len(sys.argv) > 1 else True
    _, results_path = init_directories("./data/exp_to/", test_run=test_run, min_id=0)
    device = "cuda" if pt.cuda.is_available() else "cpu"

    if test_run:
        POP_SIZE = 2
        N_EVALS = 1
        MAX_FRAMES = 3
        N_GENERATIONS = 2
    
    bs = POP_SIZE * 2
    initial_population = load_initial_population(RESULTS_PATH, bs).to(device=device)
    solution_length = initial_population.shape[1]

    n_threads = max([i for i in range(1, max(4, os.cpu_count() // 4) + 1) if bs % i == 0])


    try:
        model = instanciate_model(
            MODEL_PATH,
            input_spike_interval=INPUT_SPIKE_INTERVAL,
            batch_size=bs,
            device=device,
        )
        env = LLWrapper(bs, n_threads, device)

        lower = float(pt.amin(initial_population).item())
        upper = float(pt.amax(initial_population).item())
        span = max(1.0, upper - lower)
        pad = 0.1 * span
        initial_bounds = (lower - pad, upper + pad)

        problem = Problem(
            "max",
            make_objective(model, env, N_EVALS, MAX_FRAMES, device),
            solution_length=solution_length,
            initial_bounds=initial_bounds,
            dtype=pt.float32,
            device=device,
            vectorized=True,
        )

        searcher = Cosyne(
            problem,
            popsize=POP_SIZE,
            tournament_size=2,
            mutation_stdev=0.6,
            mutation_probability=0.1,
            permute_all=False,
            num_elites=2,
            eta=1,
            num_children=POP_SIZE-2 # needs to be even, subtract num_elites
        )

        seeded_population = problem.generate_batch(bs, empty=True)
        seeded_population.set_values(initial_population)
        problem.evaluate(seeded_population)
        searcher._population = seeded_population.take_best(POP_SIZE)
        searcher._first_generation = False


        backend = HDF5Backend(results_path + "/results.h5", compression="gzip", compression_opts=4, dtype="float32")
        create_fields(backend, POP_SIZE, solution_length)

        ctx = Context(searcher=searcher, generation=0)
        logger = Logger(backend)
        logger.add("after_step", write_population)

        logger.emit("after_step", ctx)

        best_initial = searcher.population.take_best(1).access_evals(0)[0].item()
        print(f"Initial best: {best_initial:.6f}")

        t_start = time()
        for generation in range(N_GENERATIONS):
            generation_start = time()
            searcher.step()

            ctx.generation = generation + 1
            logger.emit("after_step", ctx)

            population_evals = searcher.population.access_evals()
            best_eval = pt.max(population_evals).item()
            mean_eval = pt.mean(population_evals).item()
            median_eval = pt.median(population_evals).item()
            print(f"ep{generation}:", np.round(time() - generation_start, 2), "sec", end=";   ")
            print("mean:", round(mean_eval, 2), "median:", round(median_eval, 2), "best:", round(best_eval, 2))

        print(f"Total time: {time() - t_start:.2f}s")
    finally:
        model.chip.close()