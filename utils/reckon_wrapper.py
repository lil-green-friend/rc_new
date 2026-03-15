import torch as pt
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from utils.RC_torch import RECKON



class RECKON_Wrapper(nn.Module):
    """Allows setting parameters from float values"""
    def __init__(self, *reckon_args, **reckon_kw):
        super().__init__()
        object.__setattr__(self, "net", RECKON(*reckon_args, **reckon_kw)) # Bypass nn.Module registration
        self.net.h_max = pt.tensor([0, 127]) # adjust hidden state to uint7
        self.net.w_max = pt.tensor([-7, 7])  # adjust weights to int4
    
        self.register_buffer("n_res", pt.tensor(self.net._N) )
        self.register_buffer("n_out", pt.tensor(self.net.n_out))
        self.register_buffer("n_rec", self.n_res - self.n_out)

        self.tau = nn.Parameter(pt.empty((self.net._bs)))
        self.thr = nn.Parameter(pt.empty((self.net._bs, 4)))
        self.Wout = nn.Parameter(pt.empty((self.net._bs, self.n_out, self.n_rec))) # only positive weights, sign is handled by mask_EI

        Win = pt.zeros(self.net.Win[0].shape)
        Win[self.n_out:, :] = pt.tensor(self.net.Wrec[0, self.n_out:, self.n_out:])
        self.net.set_weights(Win, 'in', overwrite=True, respect_sparsity=False)
        self.register_buffer("Wrec", pt.tensor(self.net.Wrec[0]))
        self.register_buffer("mask_EI", self.net._mask_EI)
    
    def forward(self, x):
        # return self.net.run(poisson_inputs = x)[0]
        return pt.stack([self.net.forward(poisson_inputs=x[:, i, :]) for i in range(x.shape[1])], dim=-1)

    def set_parametrizations(self):
        parametrize.register_parametrization(self, "tau", UpdateTau(self))
        parametrize.register_parametrization(self, "thr", UpdateThr_OutRec_ExIn(self))
        parametrize.register_parametrization(self, "Wout", UpdateWout(self))
        parametrize.register_parametrization(self, "Wrec", UpdateWrec(self))


class UpdateTau(nn.Module):
    def __init__(self, base):
        super().__init__()
        object.__setattr__(self, "base", base)

    def forward(self, x):
        self.base.net.tau = x[:, None]
        return x


class UpdateThr_OutRec_ExIn(nn.Module):
    """4 Thresholds for output and recurrent neurons, excitatory and inhibitory neurons"""
    def __init__(self, base):
        super().__init__()
        object.__setattr__(self, "base", base)
    
    def forward(self, x):
        t_out_ex = x[..., 0:1]
        t_res_ex = x[..., 1:2]
        t_out_inh = x[..., 2:3]
        t_res_inh = x[..., 3:4]
        inh_mask = (self.base.mask_EI == -1).repeat(self.base.net._bs, 1)
        t = pt.hstack([t_out_ex.repeat(1, self.base.n_out), t_res_ex.repeat(1, self.base.n_rec)])
        t[inh_mask] = pt.hstack([t_out_inh.repeat(1, self.base.n_out), t_res_inh.repeat(1, self.base.n_rec)])[inh_mask]
        self.base.net.threshold = t
        return x

class UpdateWin(nn.Module):
    def __init__(self, base):
        super().__init__()
        object.__setattr__(self, "base", base)

    def forward(self, x): 
        x = pt.clip(x.round(), -7, 7)
        self.base.net.set_weights(x, 'in', overwrite=True, respect_sparsity=False)
        return x


class UpdateWout(nn.Module):
    def __init__(self, base):
        super().__init__()
        object.__setattr__(self, "base", base)

    def forward(self, x):
        Wrec = pt.full(self.base.net.Wrec.shape, pt.nan)
        Wrec[:, :self.base.n_out, self.base.n_out:] = x.reshape(-1, self.base.n_out, self.base.n_rec)
        Wrec = pt.clip(Wrec.round() * self.base.mask_EI[None], -7, 7)
        self.base.net.set_weights(Wrec, 'rec', overwrite=False, respect_sparsity=False)
        return x

class UpdateWrec(nn.Module):
    def __init__(self, base):
        super().__init__()
        object.__setattr__(self, "base", base)

    def forward(self, x):
        x[:self.base.n_out, self.base.n_out:] = pt.nan
        Wrec = x.unsqueeze(0).repeat(self.base.net._bs, 1, 1)
        Wrec = pt.clip(Wrec.round() * self.base.mask_EI[None], -7, 7)
        self.base.net.set_weights(Wrec, 'rec', overwrite=False, respect_sparsity=False)

        Win = pt.zeros(self.base.net.Win[0].shape)
        Win[self.base.n_out:, :] = pt.tensor(self.base.net.Wrec[0, self.base.n_out:, self.base.n_out:])
        self.base.net.set_weights(Win, 'in', overwrite=True, respect_sparsity=False)
        return x
    
        