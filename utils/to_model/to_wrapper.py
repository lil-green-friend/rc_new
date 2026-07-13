import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import tinyodin as to
from .sim_pool import SimPool


def _forward_batch(sim, x, n_res, input_spike_interval):
    outputs = []
    for i in range(x.shape[0]):
        spike_idx = np.nonzero(x[i])[0] + n_res
        sim.send_neuron_events_equally_spaced(
            spike_idx,
            sim.get_time() + input_spike_interval * len(spike_idx),
            input_spike_interval,
        )
        sim.all_neuron_leakage_event()
        sim.cycle_clk(512)
        spike_list = sim.read_output_spikes()
        counts = np.bincount(spike_list, minlength=256)
        outputs.append(counts)
    return np.stack(outputs, axis=-1)


class TinyOdin_Wrapper(nn.Module):
    """Allows setting parameters from float values"""
    def __init__(self, batch_size, n_res, n_out, n_actions, n_in, input_spike_interval, device, n_cpu_cores=None):
        super().__init__()
        self.bs = batch_size
        self.n_actions = n_actions
        self.device = device
        self.n_cpu_cores = max(1, min(self.bs, self.bs if n_cpu_cores is None else int(n_cpu_cores)))
        # sparsity = [self.reservoir_sparsity.item()],
        # mask_sparsity = create_sparsity_mask(self.reservoir_n.item(), self.reservoir_sparsity.item()),
        # uniform_weights = True,

        self.register_buffer("n_in", n_in)
        self.register_buffer("n_res", n_res)
        self.register_buffer("n_out", n_out)
        self.register_buffer("n_rec", self.n_res - self.n_out)
        self.register_buffer("input_spike_interval", input_spike_interval)
        self.sim_pool = SimPool(self.bs, self.init_sim, _forward_batch, n_res=self.n_res.item(), input_spike_interval=self.input_spike_interval.item())

        self.leak = nn.Parameter(pt.empty((self.bs,)))
        self.thr = nn.Parameter(pt.empty((self.bs, 2)))
        self.Win_scale = nn.Parameter(pt.empty((self.bs,)))
        object.__setattr__(self, "_Win_scale", pt.empty((self.bs,)))
        self.Win_bias = nn.Parameter(pt.empty((self.bs,)))
        object.__setattr__(self, "_Win_bias", pt.empty((self.bs,)))
        self.Wout = nn.Parameter(pt.empty((self.bs, self.n_out, self.n_rec))) # only positive weights, sign is handled by mask_EI

        self._Wrec = pt.zeros((self.bs, 256, 256), dtype=pt.int8)
        Win_0 = pt.distributions.Beta(0.5, 0.5).sample((self.n_res, self.n_in)) * 2 - 1
        Win_0[..., :self.n_out, :] = 0
        self.register_buffer("Win_0", Win_0)        
        self.register_buffer("Wrec", pt.empty(self.n_res, self.n_res))
        self.register_buffer("mask_EI", pt.empty((self.n_res,)))

    def forward(self, x):
        batch_outputs = self.sim_pool.map("forward", x.cpu().numpy())
        batch_outputs = [pt.from_numpy(x).to(self.device) for x in batch_outputs] 
        return pt.stack(batch_outputs, dim=0).clamp(max=1)

    def set_parametrizations(self):
        parametrize.register_parametrization(self, "leak", UpdateLeak(self))
        parametrize.register_parametrization(self, "thr", UpdateThr_OutRec(self))
        parametrize.register_parametrization(self, "Win_scale", UpdateWin(self, "scale"))
        parametrize.register_parametrization(self, "Win_bias", UpdateWin(self, "bias"))
        parametrize.register_parametrization(self, "Wout", UpdateWout(self))
        parametrize.register_parametrization(self, "Wrec", UpdateWrec(self))

    def close(self):
        self.sim_pool.close()
        
    def _update_neuron_memory(self, neuron_idx=None, disable=None, leak_str=None, thr=None, core=None):
        if neuron_idx is None:
            neuron_idx = pt.arange(self.n_res).unsqueeze(0).repeat(self.bs, 1).cpu().numpy()
        kwargs = {k: v.cpu().numpy() for k, v in zip(["disable", "leak_str", "thr", "core"], [disable, leak_str, thr, core]) if v is not None}
        self.sim_pool.map("write_neuron_memory", neuron_idx, **kwargs)

    def _update_synapse_memory(self, w):
        self.sim_pool.map("set_synapse_memory", w.cpu().numpy())

    @staticmethod
    def init_sim(spi_open_loop=False, spi_aer_src_ctrl_nneur=False, spi_max_neur=255):
        sim = to.Simulator()
        # ensure 
        sim.set_sck(0)
        sim.set_mosi(0)
        sim.set_aerin_addr(0)
        sim.set_aerin_req(0)
        sim.set_aerout_ack(0)
        sim.reset()
        sim.cycle_clk(100)

        sim.set_config(to.SpiConfigRegisterIndex.SPI_OPEN_LOOP, True)
        sim.write_neuron_memory(np.arange(256), thr=np.ones(256), core=np.zeros(256))
        sim.cycle_clk(1000)
        sim.set_neuron_core(np.zeros(256))
        
        # reset to stop it from spiking
        sim.read_output_spikes() # consume any spikes generated before the reset
        # set configs
        sim.set_config(to.SpiConfigRegisterIndex.SPI_OPEN_LOOP, spi_open_loop)
        sim.set_config(to.SpiConfigRegisterIndex.SPI_AER_SRC_CTRL_nNEUR, spi_aer_src_ctrl_nneur)
        sim.set_config(to.SpiConfigRegisterIndex.SPI_MAX_NEUR, spi_max_neur)
        return sim


class UpdateLeak(nn.Module):
    def __init__(self, base):
        super().__init__()
        object.__setattr__(self, "base", base)

    def forward(self, x):
        leak = x[:, None].repeat(1, self.base.n_res)
        self.base._update_neuron_memory(leak_str=leak)
        return x


class UpdateThr_OutRec(nn.Module):
    def __init__(self, base):
        super().__init__()
        object.__setattr__(self, "base", base)
    
    def forward(self, x):
        t_out = x[..., 0:1]
        t_res = x[..., 1:2]
        thr = pt.hstack([t_out.repeat(1, self.base.n_out), t_res.repeat(1, self.base.n_rec)])
        self.base._update_neuron_memory(thr=thr)
        return x

class UpdateWin(nn.Module):
    def __init__(self, base, type):
        super().__init__()
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "type", type)

    def forward(self, x): 
        if self.type=="scale":
            self.base._Win_scale = x
        else:
            self.base._Win_bias = x

        device = self.base.device
        
        scale = self.base._Win_scale[..., None, None].to(device)
        bias = self.base._Win_bias[..., None, None].to(device)
        Win_0 = self.base.Win_0[None, self.base.n_out:, :].to(device)
        Win = (Win_0 * scale + bias)
        Win = pt.clip(Win.round(), -7, 7)
        Win = Win.to(pt.int8)
        self.base._Wrec[:, self.base.n_res:self.base.n_res+self.base.n_in,self.base.n_out:self.base.n_res] = Win.transpose(1, 2)
        self.base._update_synapse_memory(self.base._Wrec)
        return x

class UpdateWout(nn.Module):
    def __init__(self, base):
        super().__init__()
        object.__setattr__(self, "base", base)

    def forward(self, x):
        Wout = pt.abs(x)*self.base.mask_EI[None, self.base.n_out:]
        Wout = pt.clip(Wout.round(), -7, 7)
        Wout = Wout.to(pt.int8)
        self.base._Wrec[:, self.base.n_out:self.base.n_res,:self.base.n_out] = Wout.transpose(1, 2)
        self.base._update_synapse_memory(self.base._Wrec)
        return x

class UpdateWrec(nn.Module):
    def __init__(self, base):
        super().__init__()
        object.__setattr__(self, "base", base)

    def forward(self, x):
        Wres = pt.abs(x) * self.base.mask_EI
        Wres = pt.clip(Wres.round(), -7, 7)
        Wres = Wres.to(pt.int8).T[None].repeat(self.base.bs, 1, 1)
        # Wres[:, :self.base.n_out, :self.base.n_out] = 0
        Wres[:, self.base.n_out:,:self.base.n_out] = self.base._Wrec[:, self.base.n_out:self.base.n_res,:self.base.n_out]
        self.base._Wrec[:, :self.base.n_res, :self.base.n_res] = Wres
        self.base._update_synapse_memory(self.base._Wrec)
        return x