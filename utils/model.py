import inspect
import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from .reckon_wrapper import RECKON_Wrapper
from abc import ABC, abstractmethod

pt.autograd.set_grad_enabled(False)

# class MyModule(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return x

class IntegerConstraint(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return pt.round(x)

class BoundedConstraint_Linear(nn.Module):
    def __init__(self, lower, upper):
        super().__init__()
        self.lower = lower
        self.range = upper - lower

    def forward(self, x):
        return triangle(x + 0.5) * self.range + self.lower

class BoundedConstraint_Cos(nn.Module):
    def __init__(self, lower, upper):
        super().__init__()
        self.lower = lower
        self.range = upper - lower

    def forward(self, x):
        return (pt.cos(x) + 1) / 2 * self.range + self.lower

class LowerBoundedConstraint_Linear(nn.Module):
    def __init__(self, lower):
        super().__init__()
        self.lower = lower
    
    def forward(self, x):
        return pt.abs(x) + self.lower

class FixedPointConstraint(nn.Module):
    def __init__(self, n_fractional_bits):
        super().__init__()
        self.n_fractional_bits = n_fractional_bits
        self.scaling_factor = 2**self.n_fractional_bits

    def forward(self, x):
        return pt.round(x * self.scaling_factor) / self.scaling_factor

def triangle(x: pt.Tensor):
    """
    realizes a periodic triangle function, osscilating between 0 and 1 with a period of 2
    it starts with a minimum at [0, 0]
    """
    return 2 * pt.abs(x/2 - pt.floor(0.5 + x/2))

class PoissonSpikeGeneration(nn.Module):
    def __init__(self, n_sub_steps):
        """Converts float point to series of n_sub_steps spike/non-spike events"""
        super().__init__()
        self.n_sub_steps = n_sub_steps

    def forward(self, x):
        x = pt.poisson(x.unsqueeze(1).expand(-1, self.n_sub_steps, -1))
        x[x > 1] = 1
        return x.bool()

class AdaptiveSpikeGeneration(nn.Module, ABC):
    def __init__(self, batch_size, n_sub_steps, n_fractional_bits):
        """Converts float point to series of n_sub_steps spike/non-spike events
        """
        super().__init__()
        self.n_sub_steps = n_sub_steps
        self.bs = batch_size
        self.target_input_activity = nn.Parameter(pt.empty(self.bs))
        self.input_activity_tau = nn.Parameter(pt.empty(self.bs))
        self.input_activity_bias_init = nn.Parameter(pt.empty(self.bs))
        self.fixed_point_constraint = FixedPointConstraint(n_fractional_bits)
        self.reset()

    def forward(self, x):
        x = x.unsqueeze(1).expand(-1, self.n_sub_steps, -1).clone()
        x = self.fixed_point_constraint(x)
        for i in range(x.shape[1]):
            x[:, i] += self.input_activity_bias.unsqueeze(1)
            x[:, i] = pt.clamp(x[:, i], min=0.0)
            x[:, i] = self.generate(x[:, i])
            self.input_activity_bias += self.fixed_point_constraint(1/self.input_activity_tau * (self.target_input_activity - pt.sum(x[:, i], dim=1)))

        return x.bool()
    
    def reset(self, mask=None):
        if mask is None:
            self.input_activity_bias = self.input_activity_bias_init.data
        else:
            self.input_activity_bias[mask] = self.input_activity_bias_init.data[mask]

    @staticmethod
    @abstractmethod
    def generate(x_i):
        pass

class AdaptivePoissonSpikeGeneration(AdaptiveSpikeGeneration):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    @staticmethod
    def generate(x_i):
        return pt.clamp(pt.poisson(x_i), max=1)

class AdaptiveUniformSpikeGeneration(AdaptiveSpikeGeneration):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
    
    @staticmethod
    def generate(x_i):
        return pt.rand_like(x_i) < x_i


class SOM_layer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.register_buffer("input_means", pt.empty((in_features,)))
        self.register_buffer("input_stds", pt.empty((in_features,)))
        self.register_buffer("weights", pt.empty((out_features, in_features)))

    def forward(self, x):
        x = (x - self.input_means) / self.input_stds
        x = x @ self.weights.T
        return x

class MyModel(nn.Module):
    def __init__(self, batch_size, device, n_inputs, n_actions, steps_per_frame, reservoir_n_in=128, reservoir_n=256, reservoir_n_out=32, reservoir_sparsity=0.8):
        super().__init__()

        buffers = ["n_inputs", "n_actions", "steps_per_frame", "reservoir_n_in", "reservoir_n", "reservoir_n_out", "reservoir_sparsity"]
        for k in buffers:
            v = locals()[k]
            v = pt.tensor(v) if not isinstance(v, pt.Tensor) else v
            self.register_buffer(k, v)

        self.bs = batch_size

        self.beta = nn.Parameter(pt.empty((self.bs,)))
        self.input_res = nn.Parameter(pt.empty((self.bs, self.n_inputs)))

        n_fractional_bits = 2

        self.som = SOM_layer(self.n_inputs, self.reservoir_n_in)
        self.spike_gen = AdaptiveUniformSpikeGeneration(self.bs, self.steps_per_frame, n_fractional_bits)
        self.chip = RECKON_Wrapper(self.bs,
            n_channels_in=1,
            n_reservoir=self.reservoir_n.item(),
            n_channels_out=self.n_actions.item(),
            res_input = self.reservoir_n_in.item(),
            res_readout = (self.reservoir_n_out // self.n_actions).item(),
            sparsity = [self.reservoir_sparsity.item()],
            mask_sparsity = create_sparsity_mask(self.reservoir_n.item(), self.reservoir_sparsity.item()),
            stochastic_rounding = True,
            constant_leak = True,
            uniform_weights = True,
            device = device,
            )
        
        parametrize.register_parametrization(self, "beta", BoundedConstraint_Linear(1, 20))
        parametrize.register_parametrization(self, "input_res", LowerBoundedConstraint_Linear(0))
        parametrize.register_parametrization(self.spike_gen, "target_input_activity", BoundedConstraint_Linear(6, 26))
        parametrize.register_parametrization(self.spike_gen, "target_input_activity", FixedPointConstraint(n_fractional_bits))
        parametrize.register_parametrization(self.spike_gen, "input_activity_tau", BoundedConstraint_Linear(4, 200))
        parametrize.register_parametrization(self.spike_gen, "input_activity_tau", FixedPointConstraint(n_fractional_bits))
        parametrize.register_parametrization(self.spike_gen, "input_activity_bias_init", BoundedConstraint_Linear(-6, 6))
        parametrize.register_parametrization(self.spike_gen, "input_activity_bias_init", FixedPointConstraint(n_fractional_bits))
        parametrize.register_parametrization(self.chip, "leak", BoundedConstraint_Linear(0, 32))
        parametrize.register_parametrization(self.chip, "leak", IntegerConstraint())
        parametrize.register_parametrization(self.chip, "thr", BoundedConstraint_Linear(1, 127))
        parametrize.register_parametrization(self.chip, "thr", IntegerConstraint())
        parametrize.register_parametrization(self.chip, "Win_scale", BoundedConstraint_Linear(0, 7))
        parametrize.register_parametrization(self.chip, "Win_bias", BoundedConstraint_Linear(-7, 7))
        parametrize.register_parametrization(self.chip, "Wout", BoundedConstraint_Cos(0, 7))
        self.chip.set_parametrizations()

        self.to(device)

    def update_parametrizations(self):
        self.beta
        self.input_res
        self.chip.leak
        self.chip.thr
        self.chip.Win_scale
        self.chip.Win_bias
        self.chip.Wout
        self.chip.Wrec
        self.spike_gen.reset()


    def forward(self, x):
        term = x[1]
        x = x[0]
        x = pt.clip(x * self.input_res, -1, 1)
        x = self.som(x)
        # x = nn.functional.normalize(pt.clamp(x + 1, min=0.0), 1, dim=1)
        x = self.spike_gen(x)
        activity = self.chip(x.to(self.chip.net.precision))
        x = activity[..., :self.reservoir_n_out, :].view(self.bs, self.n_inputs, self.n_actions, -1)
        x = pt.mean(x, dim=(1, 3), dtype=float) * (20 / self.steps_per_frame) # rate coding
        action = pt.distributions.Categorical(logits=self.beta[:, None] * x).sample() # softmax
        self.spike_gen.reset(term)
        return action, activity

def create_sparsity_mask(size, sparsity):
    sparsity = pt.from_numpy(regular_mask(size, size, sparsity)[0])
    sparsity *= (1 - pt.eye(size).long()) # no connections along diagonal
    return sparsity 

def regular_mask(rows, cols, p):
    # Create an empty mask
    mask = np.zeros((rows, cols), dtype=int)

    # Calculate the period of the pattern
    # We'll use the inverse of p rounded to the nearest integer
    period = max(2, round(1 / (1 - p)))

    # Create a base pattern for a single period
    base_pattern = np.zeros(period, dtype=int)
    ones_count = max(1, round((1 - p) * period))
    base_pattern[:ones_count] = 1
    np.random.shuffle(base_pattern)

    # Fill the mask
    for c in range(cols):
        # Determine the offset for this column
        offset = c % period

        # Create the column pattern by rolling the base pattern
        col_pattern = np.roll(base_pattern, offset)

        # Tile the pattern to fill the column
        mask[:, c] = np.tile(col_pattern, rows // period + 1)[:rows]

    # Calculate the actual sparsity
    actual_sparsity = np.mean(mask)

    return mask, actual_sparsity


def get_n_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters()) // model.bs

def get_param_offset_and_size(model: nn.Module):
    offset = 0
    d = {}
    for n, p in model.named_parameters():
        p_size = p.numel() // model.bs
        n = n.replace(".original", "")
        n = n.replace("parametrizations.", "")
        d[n] = (offset, p_size)
        offset += p_size

    return d

def set_params(model, vector, bs):
    pointer = 0
    for p in model.parameters():
        num = p.numel() // bs
        p.data = vector[:, pointer:pointer+num].view_as(p)
        pointer += num

    model.update_parametrizations()


def instanciate_model(source_path, batch_size=1, device="cpu"):
    state_dict = pt.load(source_path, map_location=pt.device(device))
    # Filter sd to only include keys that match MyModel's __init__ parameters
    model_init_params = inspect.signature(MyModel.__init__).parameters
    filtered_sd = {k: v for k, v in state_dict.items() if k in model_init_params}

    m = MyModel(batch_size, device, **filtered_sd)
    m.load_state_dict(state_dict, strict=False)
    return m