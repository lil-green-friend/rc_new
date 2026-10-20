from contextlib import AbstractContextManager

import h5py
import torch as pt


class Monitor(AbstractContextManager):
    def __init__(self, model, specs, h5_path, to_cpu=True):
        """
        specs: dict[name] = {
            "module": "som" | "spike_gen" | "chip" | "" (root model),
            "kind": "pre" | "post",
            "fn": callable(module, args, output_or_none) -> tensor_or_dict
        }
        h5_path: str, path to HDF5 file to write to
        """
        self.model = model
        self.specs = specs
        self.h5_path = h5_path
        self.to_cpu = to_cpu
        self.handles = []
        self.h5_file = None
        self.datasets = {}

    def _resolve_module(self, path):
        if path in ("", None):
            return self.model
        m = self.model
        for p in path.split("."):
            m = getattr(m, p)
        return m

    def _prep(self, x):
        if pt.is_tensor(x):
            x = x.detach()
            if self.to_cpu:
                x = x.cpu()
            return x.clone().numpy()
        return x

    def _store(self, key, value):
        if isinstance(value, dict):
            for k, v in value.items():
                sub_key = f"{key}/{k}"
                arr = self._prep(v)
                if sub_key not in self.datasets:
                    self.datasets[sub_key] = self.h5_file.create_dataset(
                        sub_key,
                        (0, *arr.shape),
                        dtype=arr.dtype,
                        maxshape=(None, *arr.shape),
                    )
                ds = self.datasets[sub_key]
                ds.resize(ds.shape[0] + 1, axis=0)
                ds[-1] = arr
        else:
            arr = self._prep(value)
            if key not in self.datasets:
                self.datasets[key] = self.h5_file.create_dataset(
                    key, (0, *arr.shape), dtype=arr.dtype, maxshape=(None, *arr.shape)
                )
            ds = self.datasets[key]
            ds.resize(ds.shape[0] + 1, axis=0)
            ds[-1] = arr

    def __enter__(self):
        self.h5_file = h5py.File(self.h5_path, "w")

        for key, spec in self.specs.items():
            module = self._resolve_module(spec.get("module", ""))
            kind = spec.get("kind", "post")
            fn = spec["fn"]

            if kind == "pre":

                def _pre_hook(mod, args, _key=key, _fn=fn):
                    self._store(_key, _fn(mod, args, None))

                h = module.register_forward_pre_hook(_pre_hook)
            else:

                def _post_hook(mod, args, out, _key=key, _fn=fn):
                    self._store(_key, _fn(mod, args, out))

                h = module.register_forward_hook(_post_hook)
            self.handles.append(h)
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()

        if self.h5_file is not None:
            self.h5_file.close()
        return False
