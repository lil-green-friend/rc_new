import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import tinyodin as to

class TinyOdin_Wrapper(nn.Module):
    """Allows setting parameters from float values"""
    def __init__(self, batch_size, n_res, n_out, n_actions, n_in, input_spike_interval, device):
        super().__init__()
        self.bs = batch_size
        assert self.bs == 1 # only have one simulator simultaneously
        self.n_actions = n_actions
        self.device = device,
        # sparsity = [self.reservoir_sparsity.item()],
        # mask_sparsity = create_sparsity_mask(self.reservoir_n.item(), self.reservoir_sparsity.item()),
        # uniform_weights = True,

        self.register_buffer("n_in", n_in)
        self.register_buffer("n_res", n_res)
        self.register_buffer("n_out", n_out)
        self.register_buffer("n_rec", self.n_res - self.n_out)
        self.register_buffer("input_spike_interval", input_spike_interval)
        self.sim = self.init_sim(spi_max_neur=n_res-1)  
        self.sim.reset()

        self.leak = nn.Parameter(pt.empty((self.bs)))
        self.thr = nn.Parameter(pt.empty((self.bs, 2)))
        self.Win_scale = nn.Parameter(pt.empty((self.bs, 2)))
        self.Wout = nn.Parameter(pt.empty((self.bs, self.n_out, self.n_rec))) # only positive weights, sign is handled by mask_EI

        self._Wrec = pt.zeros((self.bs, 256, 256), dtype=pt.int8)
        Win_0 = pt.distributions.Beta(0.5, 0.5).sample((self.n_res, self.n_in)) * 2 - 1
        Win_0[..., :self.n_out, :] = 0
        self.register_buffer("Win_0", Win_0)        
        self.register_buffer("Wrec", pt.empty(self.n_res, self.n_res))
        self.register_buffer("mask_EI", pt.empty((self.n_res,)))

    def forward(self, x):
        outputs = []
        for i in range(x.shape[1]):
            inputs = x[0, i, :]
            spike_idx = pt.nonzero(inputs) + self.n_res
            spike_idx = spike_idx.squeeze(-1).cpu().numpy()
            self.sim.send_neuron_events_equally_spaced(spike_idx, self.sim.get_time() + 
                                                       self.input_spike_interval * len(spike_idx), 
                                                       self.input_spike_interval)
            self.sim.all_neuron_leakage_event()
            spike_list = pt.from_numpy(self.sim.read_output_spikes()).to(pt.long)
            counts = pt.bincount(spike_list, minlength=256).to(pt.int8)
            outputs.append(counts)
        return pt.stack(outputs, dim=-1)[None].clamp(max=1)

    def set_parametrizations(self):
        parametrize.register_parametrization(self, "leak", UpdateLeak(self))
        parametrize.register_parametrization(self, "thr", UpdateThr_OutRec(self))
        parametrize.register_parametrization(self, "Win_scale", UpdateWin(self))
        parametrize.register_parametrization(self, "Wout", UpdateWout(self))
        parametrize.register_parametrization(self, "Wrec", UpdateWrec(self))

    def init_sim(self, spi_open_loop=False, spi_aer_src_ctrl_nneur=False, spi_max_neur=255):
        sim = to.Simulator()
        # ensure 
        sim.set_sck(0)
        sim.set_mosi(0)
        sim.set_aerin_addr(0)
        sim.set_aerin_req(0)
        sim.set_aerout_ack(0)
        sim.reset()
        sim.cycle_clk(100)
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
        neuron_idx = pt.arange(self.base.n_res)
        leak = x[0].repeat(self.base.n_res)
        self.base.sim.write_neuron_memory(neuron_idx.cpu().numpy(), leak_str=leak.cpu().numpy())
        return x


class UpdateThr_OutRec(nn.Module):
    def __init__(self, base):
        super().__init__()
        object.__setattr__(self, "base", base)
    
    def forward(self, x):
        t_out = x[0, 0:1]
        t_res = x[0, 1:2]
        neuron_idx = pt.arange(self.base.n_res)
        thr = pt.hstack([t_out.repeat(self.base.n_out), t_res.repeat(self.base.n_rec)])
        self.base.sim.write_neuron_memory(neuron_idx.cpu().numpy(), thr=thr.cpu().numpy())
        return x

class UpdateWin(nn.Module):
    def __init__(self, base):
        super().__init__()
        object.__setattr__(self, "base", base)

    def forward(self, x): 
        scale = x[0, 0][..., None, None]
        bias = x[0, 1][..., None, None]
        Win = (self.base.Win_0[None, self.base.n_out:, :] * scale + bias)
        Win = pt.clip(Win.round(), -7, 7)
        Win = Win.to(pt.int8)
        self.base._Wrec[:, self.base.n_res:self.base.n_res+self.base.n_in,self.base.n_out:self.base.n_res] = Win.transpose(1, 2)
        self.base.sim.set_synapse_memory(self.base._Wrec[0].cpu().numpy())
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
        self.base.sim.set_synapse_memory(self.base._Wrec[0].cpu().numpy())
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
        self.base.sim.set_synapse_memory(self.base._Wrec[0].cpu().numpy())
        return x