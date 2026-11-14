import torch as pt
from torch import nn

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
