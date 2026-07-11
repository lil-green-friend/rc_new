import multiprocessing as mp
import traceback

def _run_worker(pipe, init_fn, run_batch_fn, n_res, input_spike_interval):
    sim = init_fn(spi_max_neur=n_res - 1)

    while True:
        message = pipe.recv()
        if message is None:
            break

        fn_name, args, kwargs = message
        try:
            if fn_name == "forward":
                result = run_batch_fn(sim, *args, n_res, input_spike_interval)
            else:
                fn = getattr(sim, fn_name)
                result = fn(*args, **kwargs)
            pipe.send((True, result))
        except Exception:
            pipe.send((False, traceback.format_exc()))

    pipe.close()


class SimPool():
    def __init__(self, bs, init_fn, run_batch_fn, n_res, input_spike_interval):
        self.bs = bs
        self.init_fn = init_fn
        self.run_batch_fn = run_batch_fn
        self.n_res = n_res
        self.input_spike_interval = input_spike_interval

        self.ctx = mp.get_context("spawn")
        self.pipes = []
        self.processes = []
        for _ in range(self.bs):
            parent_pipe, child_pipe = self.ctx.Pipe()
            process = self.ctx.Process(
                target=_run_worker,
                args=(child_pipe, self.init_fn, self.run_batch_fn, self.n_res, self.input_spike_interval),
            )
            process.start()
            child_pipe.close()
            self.pipes.append(parent_pipe)
            self.processes.append(process)

    def map(self, fn_name, *args, **kwargs):
        for i, p in enumerate(self.pipes):
            a = [x[i] for x in args]
            kw = {k: v[i] for k, v in kwargs.items()}
            p.send((fn_name, a, kw))

        results = []
        for p in self.pipes:
            ok, payload = p.recv()
            if not ok:
                raise RuntimeError(payload)
            results.append(payload)
        return results

    def close(self):
        for p in self.pipes:
            try:
                p.send(None)
            except (BrokenPipeError, OSError):
                pass
        for process, p in zip(self.processes, self.pipes):
            process.join()
            p.close()