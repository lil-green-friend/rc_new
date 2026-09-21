import os
import h5py

class Context(dict):
    def __init__(self, data=None, **kwargs):
        super().__init__()

        if data is not None:
            self.update(data)

        self.update(kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

class Trigger:
    def __call__(self, ctx) -> bool:
        raise NotImplementedError

class SustainedCondition(Trigger):
    """Once the predicate becomes true, logs for duration steps"""

    def __init__(self, predicate, duration):
        self.predicate = predicate
        self.duration = duration
        self.counter = -1
        self.triggered = False

    def __call__(self, ctx):
        if self.triggered and self.counter < self.duration:
            self.counter += 1
            return True
        elif not self.triggered and self.predicate(ctx):
            self.counter = 0 
            self.triggered = True
            return True
        return False

class HDF5Backend():
    def __init__(self, path, compression=None, compression_opts=None, dtype=None):
        self.f = h5py.File(path, "w")
        self.paths = {}
        self.compression = compression
        self.compression_opts = compression_opts
        self.dtype = dtype

    def add_path(self, path, **kw):
        self.paths[path] = {"idx": 0}

        shape = kw.get("shape")
        if shape is None:
            kw["shape"] = (0,)
            kw.setdefault("maxshape", (None,))
            kw.setdefault("chunks", True)
        elif isinstance(shape, tuple) and any(dim == 0 for dim in shape):
            kw.setdefault("maxshape", tuple(None if dim == 0 else dim for dim in shape))
            kw.setdefault("chunks", True)

        if "compression" not in kw and self.compression is not None:
            kw["compression"] = self.compression
        if "compression_opts" not in kw and self.compression_opts is not None:
            kw["compression_opts"] = self.compression_opts
        if "dtype" not in kw and self.dtype is not None:
            kw["dtype"] = self.dtype
        
        self.f.create_dataset(path, **kw)

    def write(self, path, value):
        if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
            value = value.detach().cpu().numpy()
        self.f[path][:] = value

    def write_next(self, path, value):
        if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
            value = value.detach().cpu().numpy()
        dataset = self.f[path]
        chunk_size = dataset.chunks[0]
        expected_shape = (chunk_size,) + dataset.shape[1:]
        if value.shape != expected_shape:
            raise ValueError(
                f"Expected {path} value to have shape {expected_shape}, "
                f"got {value.shape}"
            )

        idx = self.paths[path]["idx"]
        next_idx = idx + chunk_size
        if next_idx > dataset.shape[0]:
            dataset.resize((next_idx,) + dataset.shape[1:])
        dataset[idx:next_idx] = value
        self.paths[path]["idx"] = next_idx

    def write_idx(self, path, idx, value):
        if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
            value = value.detach().cpu().numpy()
        dataset = self.f[path]
        dataset[*idx] = value

class Logger:
    def __init__(self, backend):
        self.entries = {}
        self.backend = backend

    def add(self, event, getter, trigger=lambda x: True):
        if self.entries.get(event) is None:
            self.entries[event] = []
        self.entries[event].append((getter, trigger))

    def emit(self, event, context):
        for getter, trigger in self.entries.get(event, []):
            if trigger(context):
                getter(self.backend, context)

