import torch as pt
from torch import nn


def _scalar(value, dtype):
    return pt.tensor(value, dtype=dtype) if value is not None else pt.empty(())


class IntegerType(nn.Module):
    def __init__(self, n_bits: int | None = None, signed: bool | None = None):
        super().__init__()
        self.register_buffer("n_bits", _scalar(n_bits, pt.int64))
        self.register_buffer("signed", _scalar(signed, pt.bool))
        if n_bits is not None and signed is not None:
            self._validate(n_bits, signed)

    @staticmethod
    def _validate(n_bits: int, signed: bool) -> None:
        if signed:
            if n_bits <= 1:
                raise ValueError(
                    f"Received n_bits={n_bits}. A signed type with less than two bits does not make sense."
                )
        else:
            if n_bits <= 0:
                raise ValueError(
                    f"Received n_bits={n_bits}. An unsigned type with less than one bit does not make sense."
                )

    @property
    def max(self) -> int:
        n_bits = int(self.n_bits.item())
        return (
            (1 << (n_bits - 1)) - 1 if bool(self.signed.item()) else (1 << n_bits) - 1
        )

    @property
    def min(self) -> int:
        n_bits = int(self.n_bits.item())
        return -(1 << (n_bits - 1)) if bool(self.signed.item()) else 0

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._validate(int(self.n_bits.item()), bool(self.signed.item()))

    def ensure_type(self, x: pt.Tensor) -> pt.Tensor:
        return x.to(int).clip(self.min, self.max)

    def ensure_type_round(self, x: pt.Tensor) -> pt.Tensor:
            return x.round().to(int).clip(self.min, self.max)

class IntegerAdd(nn.Module):
    # Adds two integer tensors and ensures that the result fits the given dtype
    def __init__(
        self,
        r_dtype: IntegerType | None = None,
        a_lshift: int | None = 0,
        b_lshift: int | None = 0,
        r_rshift: int | None = 0,
    ):
        super().__init__()
        self.r_dtype = r_dtype
        self.register_buffer("a_lshift", _scalar(a_lshift, pt.int64))
        self.register_buffer("b_lshift", _scalar(b_lshift, pt.int64))
        self.register_buffer("r_rshift", _scalar(r_rshift, pt.int64))

    def forward(self, a: pt.Tensor, b: pt.Tensor) -> pt.Tensor:
        a = a << self.a_lshift
        b = b << self.b_lshift
        return self.r_dtype.ensure_type((a + b) >> self.r_rshift)


class IntegerMul(nn.Module):
    # Multiplies two integer tensors and ensures that the result fits the given dtype
    def __init__(
        self,
        r_dtype: IntegerType | None = None,
        a_lshift: int | None = 0,
        b_lshift: int | None = 0,
        r_rshift: int | None = 0,
    ):
        super().__init__()
        self.r_dtype = r_dtype
        self.register_buffer("a_lshift", _scalar(a_lshift, pt.int64))
        self.register_buffer("b_lshift", _scalar(b_lshift, pt.int64))
        self.register_buffer("r_rshift", _scalar(r_rshift, pt.int64))

    def forward(self, a: pt.Tensor, b: pt.Tensor) -> pt.Tensor:
        a = a << self.a_lshift
        b = b << self.b_lshift
        return self.r_dtype.ensure_type((a * b) >> self.r_rshift)


class FloatToInt(nn.Module):
    def __init__(
        self,
        r_dtype: IntegerType | None = None,
        fmin: float | None = 0.0,
        fmax: float | None = 1.0,
    ):
        super().__init__()
        self.r_dtype = r_dtype
        self.register_buffer("fmin", _scalar(fmin, pt.float32))
        self.register_buffer("fmax", _scalar(fmax, pt.float32))
        if fmin is not None and fmax is not None and fmax == fmin:
            raise ValueError("fmax and fmin must be different")

    def forward(self, x):
        if self.fmax == self.fmin:
            raise ValueError("fmax and fmin must be different")
        k = (self.r_dtype.max - self.r_dtype.min) / (self.fmax - self.fmin)
        x = ((x - self.fmin) * k) + self.r_dtype.min
        return self.r_dtype.ensure_type_round(x)


class IntegerVecMatMul(nn.Module):
    def __init__(
        self,
        r_dtype: IntegerType | None = None,
        i_dtype: IntegerType | None = None,
        v_lshift: int | None = 0,
        m_lshift: int | None = 0,
        i_rshift: int | None = 0,
        r_rshift: int | None = 0,
    ):
        super().__init__()
        self.r_dtype = r_dtype
        self.mul = IntegerMul(i_dtype, i_rshift)
        self.add = IntegerAdd(i_dtype)
        self.register_buffer("v_lshift", _scalar(v_lshift, pt.int64))
        self.register_buffer("m_lshift", _scalar(m_lshift, pt.int64))
        self.register_buffer("i_rshift", _scalar(i_rshift, pt.int64))
        self.register_buffer("r_rshift", _scalar(r_rshift, pt.int64))

    def forward(self, v: pt.Tensor, m: pt.Tensor):
        if v.ndim != m.ndim - 1:
            raise ValueError("inputs have incompatible dimensions")

        if v.shape[-1] != m.shape[-2]:
            raise ValueError("inputs have incompatible dimensions")

        v = v << self.v_lshift
        m = m << self.m_lshift

        # a, a*b
        products = self.mul(v.unsqueeze(-1), m)  # a*b, i_dtype enforced by self.mul
        result = pt.zeros_like(products.select(-2, 0))  # b
        for i in range(products.shape[-2]):
            result = self.add(result, products.select(-2, i)) # i_dtype enforced by self.add
        return self.r_dtype.ensure_type(result >> self.r_rshift)
