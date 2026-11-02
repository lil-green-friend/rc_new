import inspect
from abc import ABC, abstractmethod

import numpy as np
import torch as pt
from torch import nn
from torch.nn.utils import parametrize

from .reckon_wrapper import RECKON_Wrapper

pt.autograd.set_grad_enabled(False)


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


class PowerOfTwoConstraint(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 2 ** pt.round(x)


def triangle(x: pt.Tensor):
    """
    realizes a periodic triangle function, osscilating between 0 and 1 with a period of 2
    it starts with a minimum at [0, 0]
    """
    return 2 * pt.abs(x / 2 - pt.floor(0.5 + x / 2))


class PoissonSpikeGeneration(nn.Module):
    def __init__(self, n_sub_steps):
        """Converts float point to series of n_sub_steps spike/non-spike events"""
        super().__init__()
        self.n_sub_steps = n_sub_steps

    def forward(self, x):
        x = pt.poisson(x.unsqueeze(1).expand(-1, self.n_sub_steps, -1))
        x[x > 1] = 1
        return x.bool()


class AdaptiveSpikeGeneration(nn.Module):
    def __init__(self, batch_size, n_sub_steps, n_fractional_bits):
        super().__init__()
        self.n_sub_steps = n_sub_steps
        self.bs = batch_size
        self.bias = nn.Parameter(pt.empty(self.bs))
        self.scale = nn.Parameter(pt.empty(self.bs))
        self.fixed_point_constraint = FixedPointConstraint(n_fractional_bits)

    def forward(self, x):
        x = self.fixed_point_constraint(x)
        x = x + self.bias.unsqueeze(1)
        x = self.fixed_point_constraint(x * self.scale.unsqueeze(1))
        x = pt.clamp(x, min=0.0)
        x = x.unsqueeze(1).expand(-1, self.n_sub_steps, -1).clone()
        x = pt.rand_like(x) < x
        return x.bool()


class RealtimeAdaptiveSpikeGeneration(nn.Module, ABC):
    def __init__(self, batch_size, n_sub_steps, n_fractional_bits):
        """Converts float point to series of n_sub_steps spike/non-spike events"""
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
            self.input_activity_bias += self.fixed_point_constraint(
                1
                / self.input_activity_tau
                * (self.target_input_activity - pt.sum(x[:, i], dim=1))
            )

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


class AdaptivePoissonSpikeGeneration(RealtimeAdaptiveSpikeGeneration):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    @staticmethod
    def generate(x_i):
        return pt.clamp(pt.poisson(x_i), max=1)


class AdaptiveUniformSpikeGeneration(RealtimeAdaptiveSpikeGeneration):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    @staticmethod
    def generate(x_i):
        return pt.rand_like(x_i) < x_i


class _SOMNormalize(nn.Module):
    def forward(self, x, input_means, input_stds):
        return (x - input_means) / input_stds


class _SOMProjection(nn.Module):
    def forward(self, x, weights):
        return x @ weights.T


class SOM_layer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.register_buffer("input_means", pt.empty((in_features,)))
        self.register_buffer("input_stds", pt.empty((in_features,)))
        self.register_buffer("weights", pt.empty((out_features, in_features)))
        self.normalize = _SOMNormalize()
        self.projection = _SOMProjection()

    def forward(self, x):
        x = self.normalize(x, self.input_means, self.input_stds)
        return self.projection(x, self.weights)


class _InputScale(nn.Module):
    def forward(self, x, input_res):
        return x * input_res

class _InputClip(nn.Module):
    def forward(self, x):
        return pt.clip(x, -1, 1)

class _ReservoirReadout(nn.Module):
    def __init__(self, batch_size, n_inputs, n_actions, reservoir_n_out):
        super().__init__()
        self.batch_size = batch_size
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.reservoir_n_out = reservoir_n_out

    def forward(self, activity):
        x = activity[..., : self.reservoir_n_out, :]
        return x.view(self.batch_size, self.n_inputs, self.n_actions, -1)


class _RateCode(nn.Module):
    def __init__(self, steps_per_frame):
        super().__init__()
        self.steps_per_frame = (
            steps_per_frame.item()
            if isinstance(steps_per_frame, pt.Tensor)
            else steps_per_frame
        )

    def forward(self, x):
        return pt.mean(x, dim=(1, 3), dtype=float) * (20 / self.steps_per_frame)


class _ActionSampler(nn.Module):
    def forward(self, x, beta):
        return pt.distributions.Categorical(
            logits=beta[:, None] * x
        ).sample()


class _ActivityOutput(nn.Module):
    def forward(self, reservoir_activity, som_activity):
        return pt.cat((reservoir_activity, som_activity.transpose(1, 2)), dim=1)


class MyModel(nn.Module):
    def __init__(
        self,
        batch_size,
        device,
        n_inputs,
        n_actions,
        steps_per_frame,
        reservoir_n_in=128,
        reservoir_n=256,
        reservoir_n_out=32,
        reservoir_sparsity=0.8,
    ):
        super().__init__()

        buffers = [
            "n_inputs",
            "n_actions",
            "steps_per_frame",
            "reservoir_n_in",
            "reservoir_n",
            "reservoir_n_out",
            "reservoir_sparsity",
        ]
        for k in buffers:
            v = locals()[k]
            v = pt.tensor(v) if not isinstance(v, pt.Tensor) else v
            self.register_buffer(k, v)

        self.bs = batch_size

        self.beta = nn.Parameter(pt.empty((self.bs,)))
        self.input_res = nn.Parameter(pt.empty((self.bs, self.n_inputs)))

        self.input_scale = _InputScale()
        self.input_clip = _InputClip()
        n_fractional_bits = 2

        self.som = SOM_layer(self.n_inputs, self.reservoir_n_in)
        self.spike_gen = AdaptiveSpikeGeneration(
            self.bs, self.steps_per_frame, n_fractional_bits
        )
        self.chip = RECKON_Wrapper(
            self.bs,
            n_channels_in=1,
            n_reservoir=self.reservoir_n.item(),
            n_channels_out=self.n_actions.item(),
            res_input=self.reservoir_n_in.item(),
            res_readout=(self.reservoir_n_out // self.n_actions).item(),
            sparsity=[self.reservoir_sparsity.item()],
            mask_sparsity=create_sparsity_mask(
                self.reservoir_n.item(), self.reservoir_sparsity.item()
            ),
            stochastic_rounding=True,
            constant_leak=True,
            uniform_weights=True,
            device=device,
        )
        self.reservoir_readout = _ReservoirReadout(
            self.bs,
            self.n_inputs.item(),
            self.n_actions.item(),
            self.reservoir_n_out.item(),
        )
        self.rate_code = _RateCode(self.steps_per_frame)
        self.action_sampler = _ActionSampler()
        self.activity_output = _ActivityOutput()

        parametrize.register_parametrization(
            self, "beta", BoundedConstraint_Linear(1, 20)
        )
        parametrize.register_parametrization(
            self, "input_res", LowerBoundedConstraint_Linear(0)
        )
        parametrize.register_parametrization(
            self.spike_gen, "bias", BoundedConstraint_Linear(-2, 1.75)
        )
        parametrize.register_parametrization(
            self.spike_gen, "bias", FixedPointConstraint(n_fractional_bits)
        )
        parametrize.register_parametrization(
            self.spike_gen, "scale", BoundedConstraint_Linear(-2, 1.75)
        )
        parametrize.register_parametrization(
            self.spike_gen, "scale", FixedPointConstraint(n_fractional_bits)
        )
        parametrize.register_parametrization(
            self.chip, "leak", BoundedConstraint_Linear(0, 32)
        )
        parametrize.register_parametrization(self.chip, "leak", IntegerConstraint())
        parametrize.register_parametrization(
            self.chip, "thr", BoundedConstraint_Linear(1, 127)
        )
        parametrize.register_parametrization(self.chip, "thr", IntegerConstraint())
        parametrize.register_parametrization(
            self.chip, "Win_scale", BoundedConstraint_Linear(0, 7)
        )
        parametrize.register_parametrization(
            self.chip, "Win_bias", BoundedConstraint_Linear(-7, 7)
        )
        parametrize.register_parametrization(
            self.chip, "Wout", BoundedConstraint_Cos(0, 7)
        )
        self.chip.set_parametrizations()

        self.to(device)

    def forward(self, x):
        x = self.input_scale(x, self.input_res)
        x = self.input_clip(x)
        x = self.som(x)
        som_activity = self.spike_gen(x)
        res_activity = self.chip(som_activity.to(self.chip.net.precision))
        x = self.reservoir_readout(res_activity)
        x = self.rate_code(x)
        action = self.action_sampler(x, self.beta)
        activity = self.activity_output(res_activity, som_activity)
        return action, activity


def create_sparsity_mask(size, sparsity):
    sparsity = pt.from_numpy(regular_mask(size, size, sparsity)[0])
    sparsity *= 1 - pt.eye(size).long()  # no connections along diagonal
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
    with pt.no_grad():
        for p in model.parameters():
            num = p.numel() // bs
            values = vector[:, pointer : pointer + num].reshape_as(p)
            p.copy_(values)
            pointer += num

    cache = getattr(parametrize, "_cache", None)
    if cache is not None:
        cache.clear()


def instanciate_model(source_path, batch_size=1, device="cpu"):
    state_dict = pt.load(source_path, map_location=pt.device(device))
    # Filter sd to only include keys that match MyModel's __init__ parameters
    model_init_params = inspect.signature(MyModel.__init__).parameters
    filtered_sd = {k: v for k, v in state_dict.items() if k in model_init_params}

    m = MyModel(batch_size, device, **filtered_sd)
    m.load_state_dict(state_dict, strict=False)
    return m
