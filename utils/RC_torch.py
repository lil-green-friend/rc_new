"""
    Reservoir computing, PyTorch version for GPU acceleration

    SuperRC (RC base class)
    |--> LSM (Liquid State Machine base class)
         |--> ODIN (minimal hardware implementation)
             |--> RECKON (minimal hardware implementation with learning)
"""
import os
import numpy as np
import torch as pt

from torch import nn
from torch.distributions import Beta, Normal

import matplotlib.pyplot as mpl


PRECISION = 'half' if pt.cuda.is_available() else 'float'

pt.autograd.set_grad_enabled(False)

def is_iter(var):
    if hasattr(var, '__iter__') and not is_zero_tensor(var):
        return True
    else:
        return False
def is_zero_tensor(var):
    if isinstance(var, pt.Tensor) and var.dim() == 0:
        return True
    else:
        return False

class SuperRC(nn.Module):
    """
    Echo State Network / Liquid State Machine base class

    The state of each node (neuron) is given by its output o and hidden state h.
        o_t+1 = f_transfer( h_t+1 )
        h_t+1 = h_t + (W_rec x o_t + W_in x inputs_t) / tau

    Parameters:
        batch_size (int):
            Number of networks that are evaluated in parallel for efficient GPU use.
            Network structure is identical over a batch but parameters may differ.
            Parameters must be given as basic data types or tensors of either shape
                * (1, n_reservoir)           --> node-specific parameter
                * (batch_size, 1)            --> network-specific parameter
                * (batch_size, n_reservoir)  --> node/net-specific parameter
                * (specific value [provided in doc])
        n_input (int):
            Number of input nodes.
        n_reservoir (int):
            Number of reservoir nodes.
        n_readout (int):
            The first n neurons of the reservoir will be designated as 'readouts',
            but will still be kept in the recurrent loop.
        tau (float or tensor[param]):
            Time constant for the reservoir dynamics.
            Defaults to 10.
        transfer_function (function):
            Transfer function applied to h.
            Defaults to identity function.
        var_noise (float or tensor[param]):
            Variance (scale) of the Gaussian white noise added to reservoir activations.
            Defaults to 0 (no noise).
        n_modules (int):
            Number of internal modules of equal size.
            Defaults to None (no modularity).
        size_modules (int or list or tuple of int):
            If int: size of modules, if list/tuple: individual size of each module.
            Defaults to None (no modularity).
        sparsity (float or list of len 0..3):
            List of sparsity values for intra-, inter-modular and input connections.
            Range is [0..1] indicating the proportion of empty connections.
            Defaults to zero sparsity (full connectivity).
        mask_sparsity (list of tensor[bool] of len 2):
            Pre-defined sparsity masks for recurrent and input weight matrices, respectively.
        readout_feedback (bool):
            If False: no feedback from readout nodes.
            Defaults to True.
        direct_IO (bool):
            If False: no input connections to readout nodes.
            Defaults to True.
        monitoring_variables (tuple of str):
            Define variables eligible for monitoring.
            Defaults to: 'h': hidden, 'o': output.
        monitors (str or list of string):
            Specify which variables to monitor during training.
            If 'all': create all monitors specified in monitoring variables.
            Defaults to None (no monitors).
        device (str):
            Device for performing tensor computations: 'cpu', 'cuda' [GPU].
            Defaults to 'cuda'.
        precision (str):
            Data type for representing internal states: 'double', 'float', 'half'.
            Defaults to half precision.
    """
    def __init__(
        self,
        batch_size,
        n_input,
        n_reservoir,
        n_readout,
        tau = 10.,
        transfer_function = lambda x: x,
        var_noise = 0,
        n_modules = None,
        size_modules = None,
        sparsity = None,
        mask_sparsity = None,
        readout_feedback = True,
        direct_IO = True,
        monitoring_variables = None,
        monitors = None,
        device = 'cuda',
        precision = PRECISION
    ):

        self.batch_size = self._bs = batch_size

        # global variables
        self.device = device
        if 'cuda' in self.device and not pt.cuda.is_available():
            print("###  no GPU detected  ###")
            self.device = 'cpu'
        pt.set_default_device(self.device)  # this sets the global torch defaults!

        precision_level = {'half': pt.half,
                           'float': pt.float,
                           'double': pt.double,}[precision]
        pt.set_default_dtype(precision_level)
        self.precision = precision_level

        self._T = pt.tensor(0, dtype=pt.int)  # current iteration step

        self._f = transfer_function

        # reservoir dimensions
        self.n_input = n_input
        self._N = n_reservoir
        self.n_readout = self.n_out = n_readout

        # node parameters
        if tau is not None:
            self.tau = tau
        if var_noise is not None:
            self.var_noise = var_noise

        # network inputs
        self._inputs = None

        # initialize reservoir state
        self.h = pt.zeros((self._bs, self._N))
        self.o = pt.zeros((self._bs, self._N))

        self.h_max = pt.tensor([-100, 100])  # to avoid numerical issues

        self.monitoring_variables = ('h', 'o') if monitoring_variables is None else monitoring_variables
        self.create_monitors(monitors)

        self._create_weights()

        self._create_modules(n_modules, size_modules)

        self._sparsify(sparsity, mask_sparsity, readout_feedback, direct_IO)

    @property
    def tau(self):
        return self._tau
    @tau.setter
    def tau(self, value):
        self._tau = pt.abs(pt.tensor(value))
        self._alpha = ((self._tau - 1) / self._tau)
    @property
    def var_noise(self):
        return self._var_noise
    @var_noise.setter
    def var_noise(self, value):
        self._var_noise = pt.abs(pt.tensor(value))
        if pt.sum(self._var_noise):
            self._noise = True
            self.Normal = Normal(pt.tensor(0.0), pt.tensor(1.0))
        else:
            self._noise = False

    def create_monitors(self, monitors):
        """ create a dictionary of monitors """
        self._monitors = {}
        if monitors == 'all':
            for x in self.monitoring_variables:
                exec('self._monitors["'+ x +'"] = self._monitor_'+ x +' = []')
        elif hasattr(monitors, '__iter__'):
            for x in monitors:
                if x in self.monitoring_variables:
                    exec('self._monitors["'+ x +'"] = self._monitor_'+ x +' = []')
    @property
    def monitor_h(self):
        if hasattr(self, '_monitor_h'):
            return pt.stack(self._monitor_h).double().cpu().numpy()
    @property
    def monitor_o(self):
        if hasattr(self, '_monitor_o'):
            return pt.stack(self._monitor_o).double().cpu().numpy()

    def _create_weights(self):
        """ initialize network weight matrices as zeros """
        self._Wrec = pt.zeros((self._bs, self._N, self._N))
        self._Win = pt.zeros((self._bs, self._N, self.n_input))
        self.shape = self._Wrec.shape[1:]

    def set_weights(self, weights, weight_type='rec', overwrite=True, respect_sparsity=True):
        """
        Set network weights.

        weights (tensor):
            Weight matrices for recurrent or input weights.
            If only one matrix is given, it will be duplicated for the batch.
        weight_type (str):
            Setting either recurrent ('rec') or input ('in', 'inp') weights
        overwrite (bool):
            If False: nan-entries in 'weights' will not alter the respective weight value.
        respect_sparsity (bool):
            If True: only update weights where sparsity mask is True.

        """
        if self.precision == pt.half:
            weights = pt.tensor(weights).half()
        elif self.precision == pt.float:
            weights = pt.tensor(weights).float()
        else:
            weights = pt.tensor(weights).double()

        if weights.dim() == 2:
            weights = weights.repeat(self._bs, 1, 1)

        if weight_type == 'rec' and respect_sparsity:
            weights *= self._mask_sparsity.int()
        elif weight_type in ('in', 'inp') and respect_sparsity:
            weights *= self._mask_sparsity_in.int()

        if weight_type == 'rec':
            if overwrite:
                self._Wrec[...] = weights
            else:
                self._Wrec[~pt.isnan(weights)] = 0
                weights[pt.isnan(weights)] = 0
                self._Wrec += weights
        elif weight_type in ('in', 'inp'):
            if overwrite:
                self._Win[...] = weights
            else:
                self._Win[~pt.isnan(weights)] = 0
                weights[pt.isnan(weights)] = 0
                self._Win += weights
        else:
            assert weight_type in ('rec', 'in', 'inp')
    @property
    def Wrec(self):
            return self._Wrec.cpu().numpy()
    @property
    def Win(self):
            return self._Win.cpu().numpy()

    def _create_modules(self, n_modules=None, size_modules=None):
        """
        Each module is represented as a block on the diagonal of the recurrent weight matrix.
        This allows for example for block-specific sparsity.

        n_modules (int):
            Number of internal modules of equal size.
            Defaults to None (no modularity).
        size_modules (int or list or tuple of int):
            If int: size of modules, if list/tuple: individual size of each module.
            Defaults to None (no modularity).
        """
        self._mask_modularity = pt.zeros(self.shape, dtype=pt.int)
        # create a given number of equally sized modules
        if size_modules is None:
            self._n_modules = 1 if n_modules is None else n_modules
            self._size_modules = ms = pt.tensor([self._N // self._n_modules] * self._n_modules)
        else:
            # create modules of uniform size
            if not is_iter(size_modules):
                self._size_modules = ms = pt.tensor([size_modules] * (self._N // size_modules))
            # create modules of varying size
            else:
                self._size_modules = ms = pt.tensor(size_modules)
            self._n_modules = len(self._size_modules)
        for m in range(self._n_modules):
            self._mask_modularity[sum(ms[:m]):sum(ms[:m + 1]), sum(ms[:m]):sum(ms[:m + 1])] = m + 1
    @property
    def mask_modularity(self):
        return self._mask_modularity.cpu().numpy()

    def _sparsify(self, sparsity=None, mask_sparsity=None, readout_feedback=True, direct_IO=True):
        """
        Create a sparsity mask that disables a random percentage of synapses.

        sparsity (float or list of len 0..3):
            List of sparsity values for intra-, inter-modular and input connections.
            Range is [0..1] indicating the proportion of empty connections.
            Defaults to zero sparsity (full connectivity).
        mask_sparsity (list of tensor[bool] of len 2):
            Pre-defined sparsity masks for recurrent and input weight matrices, respectively.
        readout_feedback (bool):
            If False: no feedback from readout nodes.
            Defaults to True.
        direct_IO (bool):
            If False: no input connections to readout nodes.
            Defaults to True.

        """
        if mask_sparsity is None:
            # no entries on the main diagonal
            self._mask_sparsity = pt.ones(self.shape, dtype=pt.bool) ^ pt.eye(self.shape[0], dtype=pt.bool)
            self._mask_sparsity_in = pt.ones(self._Win[0].shape, dtype=pt.bool)
            if sparsity is not None:
                if not is_iter(sparsity):
                    sparsity = [sparsity] *3
                # intra-module sparsity
                if len(sparsity) > 0:
                    n_intra = pt.sum(self._mask_modularity > 0)
                    ids = pt.randperm(n_intra)[: int(n_intra * sparsity[0])]
                    idc = pt.nonzero(self._mask_modularity > 0)
                    idc = idc[ids].T
                    self._mask_sparsity[idc[0], idc[1]] = 0
                # inter-module sparsity
                if len(sparsity) > 1:
                    n_inter = pt.sum(self._mask_modularity == 0)
                    ids = pt.randperm(n_inter)[: int(n_intra * sparsity[1])]
                    idc = pt.nonzero(self._mask_modularity == 0)
                    idc = idc[ids].T
                    self._mask_sparsity[idc[0], idc[1]] = 0
                # input sparsity
                if len(sparsity) > 2:
                    n_in = self._mask_sparsity_in.numel()
                    ids = pt.randperm(n_in)[: int(n_in * sparsity[2])]
                    idc = pt.nonzero(self._mask_sparsity_in)
                    idc = idc[ids].T
                    self._mask_sparsity_in[idc[0], idc[1]] = 0
        else:
            if len(mask_sparsity) <= 2:
                self._mask_sparsity = pt.tensor(mask_sparsity[0]).bool()
                assert self._mask_sparsity.shape == self.shape
                if len(mask_sparsity) == 2:
                    self._mask_sparsity_in = pt.tensor(mask_sparsity[1]).bool()
                    assert self._mask_sparsity_in.shape == self._Win[0].shape
                else:
                    self._mask_sparsity_in = pt.zeros_like(self._Win[0]).bool()
            else:
                self._mask_sparsity = pt.tensor(mask_sparsity).bool()
                assert self._mask_sparsity.shape == self.shape
                self._mask_sparsity_in = pt.zeros_like(self._Win[0]).bool()
        if not readout_feedback:
            self._mask_sparsity[:, : self.n_out] = 0
        if not direct_IO:
            self._mask_sparsity_in[: self.n_out] = 0

    @property
    def mask_sparsity(self):
        return self._mask_sparsity.cpu().numpy()
    @property
    def mask_sparsity_in(self):
        return self._mask_sparsity_in.cpu().numpy()

    def reset_state(self, monitors=True):
        """ reset dynamic variables """
        self._T[()] = 0

        self.h[()] = 0
        self.o[()] = 0

        if monitors == True:
            for mon in self._monitors.values():
                del mon[:]

    def forward(self, inputs, noise=None):
        """
        Update network states.

        inputs (tensor):
            Input tensor of size (batch_size, n_input).
        noise (tensor):
            Tensor containing random noise of size (batch_size, n_reservoir).
            defaults to None: noise is computed on-the-fly

        """
        if noise is None:
            if self._noise:
                noise = self.Normal.sample((self._bs, self._N)) * self._var_noise
            else:
                noise = 0
        I_input = pt.einsum('bji,bi->bj', self._Win, inputs) + noise
        I_rec = pt.einsum('bji,bi->bj', self._Wrec, self.o)
        self.h = pt.clip( self._alpha * self.h + I_rec + I_input, *self.h_max)
        self.o = self._f( self.h )

        self._T += 1

        for name, mon in self._monitors.items():
            if name == 'h': mon.append(self.h)
            elif name == 'o': mon.append(self.o)

        return self.o

    def run(self, inputs=None, steps=None, noise=None, stateful=True):
        """
        Update network states iteratively. Meant as a convenience function for testing.
        Depending on steps and batch size this might be faster than forward.

        inputs (tensor):
            Input tensor of size (batch_size, steps, n_input).
            Defaults to zero tensor.
        steps (int):
            Number of steps to simulate.
            Defaults to input shape if input is not None else 1.
        noise (tensor):
            Tensor containing random noise of size (batch_size, steps, n_reservoir).
            defaults to None: noise is computed on-the-fly

        """
        if steps is None:
            if inputs is not None:
                steps = inputs.shape[1]
            else:
                steps = 1
        if inputs is None:
            if self._inputs is None:
                inputs = pt.zeros((self._bs, steps, self.n_input))
            else:
                len_inp = len(self._inputs)
                inputs = self._inputs[:, self._T % len_inp : (self._T + steps) % len_inp]

        # pre-compute noise
        if noise is None:
            if self._noise:
                noise = self.Normal.sample((steps, self._bs, self._N)) * self._var_noise[None, :, :]
            else:
                noise = 0

        # pre-compute inputs
        I_input = pt.einsum('bji,bti->tbj', self._Win, inputs) + noise

        # temporary state variables
        h = pt.zeros((steps + 1, self._bs, self._N))
        o = pt.zeros((steps + 1, self._bs, self._N))

        if stateful:
            h[0] = self.h
            o[0] = self.o

        # iterative update
        for t in range(steps):
            I_rec = pt.einsum('bji,bi->bj', self._Wrec, o[t])
            h[t + 1] = self._alpha * h[t] + I_rec + I_input[t]
            h[t + 1] = pt.clip(h[t + 1], *self.h_max)
            o[t + 1] = self._f(h[t + 1])

            self._T += 1

        if stateful:
            self.h = h[-1].clone()
            self.o = o[-1].clone()

        for name, mon in self._monitors.items():
            if name == 'h': mon.extend(h[1:])
            elif name == 'o': mon.extend(o[1:])

        return o[1:].cpu().numpy(), h[1:].cpu().numpy()

    def values(self, truncate=20, summary_statistics=False):
        """
            Returns a dictionary of network parameters and internal values.
            Entries longer than 'truncate' are not shown - only their mean and
            standard deviation (over all networks) if 'summary_statistics'.
        """
        V = {}
        for k, v in self.__dict__.items():
            if isinstance(v, dict) or 'monitor_' in k:
                continue
            if isinstance(v, list):
                v = pt.stack(v)
            try:
                if isinstance(v, tuple) and not 'mask' in k:
                    l = len(v)
                else:
                    l = len(v.flatten())
                if l <= truncate:
                    V[k] = v
                elif summary_statistics and isinstance(v, pt.Tensor):
                    V[k+" - mean"] = pt.mean(v.double())
                    V[k+" - sd"] = pt.std(v.double())
            except:
                V[k] = v
        return V

#%%

class LSM(SuperRC):
    """
    Liquid State Machine base class using LIF neurons

    The state of each node (neuron) is given by its binary output o and hidden state h.
        o_t+1 = step_threshold( h_t+1 )
        h_t+1 = h_t + ((rest - h_t) + W_rec x o_t + W_in x inputs_t) / tau

    Parameters:
        batch_size (int):
            Number of networks that are evaluated in parallel for efficient GPU use.
            Network structure is identical over a batch but parameters may differ.
            Parameters must be given as basic data types or tensors of either shape
                * (1, n_reservoir)           --> node-specific parameter
                * (batch_size, 1)            --> network-specific parameter
                * (batch_size, n_reservoir)  --> node/net-specific parameter
                * (specific value [provided in doc])
        n_input (int):
            Number of input nodes.
        n_reservoir (int):
            Number of reservoir nodes.
        n_channels_out (int):
            Number of readout _channels_, each with resolution res_readout.
            The first n (n_channels * res) neurons of the reservoir will be designated as 'readouts',
            but will still be kept in the recurrent loop by default (see readout_feedback).
            Defaults to None.
        res_readout (int):
            Node resolution of the readout channels.
            Defaults to 8.
        p_inh (float):
            Percentage of inhibitory neurons per module.
            Defaults to 50 %.
        tau (float or tensor[param]):
            Time constant for the reservoir dynamics.
            Defaults to 10 (ms).
        rest (float or tensor[param]):
            Resting (equilibrium) membrane potential of LIF neurons
            Defaults to -70 (mV).
        threshold (float or iterable or tensor[param]):
            Threshold potential for producing a spike event.
            * if iterable of length 2: set value for readout and core neurons separately
            * if tensor of shape (2, batch_size): set thresholds separately and uniquely per network
            Defaults to -54 (mV).
        reset (float or tensor[param]):
            After-spike reset potential.
            * network-specific only!
            Defaults to -80 (mV).
        refractory_period (int or tensor[param]):
            Period after a spike event where mebrane potential cannot be raised.
            Defaults to 3 (ms).
        var_noise (float or tensor[param]):
            Variance (scale) of the Gaussian white noise added to reservoir activations.
            Defaults to 0 (no noise).
        n_modules (int):
            Number of internal modules of equal size.
            Defaults to None (no modularity).
        size_modules (int or list or tuple of int):
            If int: size of modules, if list/tuple: individual size of each module.
            Defaults to None (no modularity).
        sparsity (float or list of len 0..3 [of lists of len 0..4]):
            List of sparsity values for intra-, inter-modular and input connections.
            Sublists define excitatory / inhibitory connections (e->e, e->i, i->e, i->i).
            Range is [0..1] indicating the proportion of empty connections for a given type.
            Defaults to no sparsity (full connectivity).
        mask_sparsity (list of tensor[bool] of len 2):
            Pre-defined sparsity masks for recurrent and input weight matrices, respectively.
        readout_feedback (bool):
            If False: no feedback from readout nodes.
            Defaults to True.
        direct_IO (bool):
            If False: no input connections to readout nodes.
            Defaults to True.
        beta (list of len 0..3 or list of lists [of lists] or 0, float):
            Beta and alpha values as well as a scaling factor for drawing weights from beta distribution.
            Up to 3 lists for intra-modular and inter-modular as well the input weight matrix.
            Sublists define excitatory / inhibitory connections (e->e, e->i, i->e, i->i).
            Defaults to uniform distribution of range 0..1.
            If 0: no weight init
        discretize_weights (bool):
            Intialize weights as int.
            Defaults to False.
        uniform_weights (bool):
            Init all networks within the batch with the same weights.
            Defaults to False.
        monitoring_variables (tuple of str):
            Define variables eligible for monitoring.
            Defaults to: 'h': hidden, 'o': output.
        monitors (str or list of string):
            Specify which variables to monitor during training.
            'h': hidden, 'o': output
            If 'all': create all monitors specified in monitoring variables.
            Defaults to None (no monitors).
        device (str):
            Device for performing tensor computations: 'cpu', 'cuda' [GPU].
            Defaults to 'cuda'.
        precision (str):
            Data type for representing internal states: 'double', 'float', 'half'.
            Defaults to half precision.
    """
    def __init__(
            self,
            batch_size,
            n_input,
            n_reservoir,
            n_channels_out = None,
            res_readout = 8,
            p_inh = 0.5,
            tau = 10.,
            rest = -70.,
            threshold = -54.,
            reset = -80.,
            refractory_period = 3,
            var_noise = 0,
            n_modules = None,
            size_modules = None,
            sparsity = None,
            mask_sparsity = None,
            readout_feedback = True,
            direct_IO = True,
            beta = None,
            discretize_weights = False,
            uniform_weights = False,
            monitoring_variables = None,
            monitors = None,
            device = 'cuda',
            precision = PRECISION
        ):
        # spiking non-linearity
        if precision == 'double':
            def f(x):
                return (x > self._threshold).double()
        elif precision == 'float':
            def f(x):
                return (x > self._threshold).float()
        else:
            def f(x):
                return (x > self._threshold).half()

        n_channels_out = n_channels_out if n_channels_out is not None else 0
        super().__init__(
                batch_size = batch_size,
                n_input = n_input,
                n_reservoir = n_reservoir,
                n_readout = res_readout * n_channels_out,
                tau = tau,
                transfer_function = f,
                var_noise = var_noise,
                n_modules = n_modules,
                size_modules = size_modules,
                monitoring_variables = monitoring_variables,
                monitors = monitors,
                device = device,
                precision = precision,
            )

        # additional spiking neuron parameters
        self.reset = pt.tensor(reset)
        self.rest = pt.tensor(rest)
        self.h[:] = self.rest
        self.threshold = threshold

        self.t_ref = pt.tensor(0 if refractory_period is None else refractory_period)
        self._ref = True if pt.sum(self.t_ref) > 0 else False
        self.t_last_spike = pt.zeros((self._bs, self._N,)) + pt.inf

        self.n_channels_out = n_channels_out
        self._create_inhibition(p_inh)

        self._sparsify(sparsity, mask_sparsity, readout_feedback, direct_IO)

        self._init_weights(beta, discretize_weights, uniform_weights)

    @property
    def threshold(self):
        return self._threshold
    @threshold.setter
    def threshold(self, value):
        if is_zero_tensor(value):
            self._threshold = value
        elif is_iter(value) and len(value) == 2:  # different thresholds for readout and reservoir
            value = pt.tensor(value)
            self._threshold = pt.concat((pt.stack([value[0]] * self.n_readout),
                                        pt.stack([value[1]] * (self._N - self.n_readout))
                                        )).T
        else:
            self._threshold = pt.tensor(value)

    def _create_inhibition(self, p_inh):
        """
        Split network into inhibitoy and excitatory neuron populations (per module)

        p_inh (float):
            percentage of inhibitory neurons per module
        """
        self._mask_EI = pt.ones(self._N)
        # number of inhibitory neurons per module
        n_inh_module = pt.ceil(self._size_modules * p_inh).int()
        # number of non-readout inhibitory neurons in module 1
        n_inh_module1 = n_inh_module[0] - int(pt.ceil(pt.tensor(self.n_readout * p_inh)))
        # number of inhibitory neurons not belongign to a module
        n_inh_rest = int((self._N - pt.sum(self._size_modules)) * p_inh)

        # tile readout populations with inhibitory neurons
        if self.n_readout > 4 * self.n_channels_out and self.n_channels_out > 0:
            self._mask_EI[:self.n_readout] = pt.tile( pt.repeat_interleave( pt.tensor([1, -1]),
                                                                          int(self.n_readout / self.n_channels_out / 2) ),
                                                    [self.n_channels_out,])
        # create other inhibitory subpopulations
        self._mask_EI[self._size_modules[0] - n_inh_module1 : self._size_modules[0]] = -1
        for m in range(1, self._n_modules):
            self._mask_EI[pt.sum(self._size_modules[:m+1]) - n_inh_module[m] : pt.sum(self._size_modules[:m+1])] = -1
        if n_inh_rest:
            self._mask_EI[-n_inh_rest:] = -1

        # recurrent connection types: e-->e, e-->i, i-->e, i-->i (3, 1, -1, -3)
        self._mask_rec = 2 * pt.tile(self._mask_EI[:,None], [1, self._N]).T +  pt.tile(self._mask_EI[:,None], [1, self._N])
    @property
    def mask_EI(self):
        return self._mask_EI.cpu().numpy()
    @property
    def mask_rec(self):
        return self._mask_rec.cpu().numpy()

    def _sparsify(self, sparsity=None, mask_sparsity=None, readout_feedback=True, direct_IO=True):
        """
        Sparsify recurrent connections. Inter-/intra-module as well as
        excitatory/inhibitory connections can be parameterized individually.

        sparsity (float or list of len 0..3 [of lists of len 0..4]):
            List of sparsity values for intra-, inter-modular and input connections.
            Sublists define excitatory / inhibitory connections (e->e, e->i, i->e, i->i).
            Range is [0..1] indicating the proportion of empty connections for a given type.
            Defaults to no sparsity (full connectivity).
        mask_sparsity (list of tensor[bool] of len 2):
            Pre-defined sparsity masks for recurrent and input weight matrices, respectively.
        readout_feedback (bool):
            If False: no feedback from readout nodes.
            Defaults to True.
        direct_IO (bool):
            If False: no input connections to readout nodes.
            Defaults to True.

        """
        if mask_sparsity is None:
            self._mask_sparsity = pt.ones(self.shape, dtype=pt.bool) ^ pt.eye(self.shape[0], dtype=pt.bool)
            self._mask_sparsity_in = pt.ones(self._Win[0].shape, dtype=pt.bool)
            if sparsity is not None:
                if not is_iter(sparsity):
                    sparsity = [sparsity] *3
                if len(sparsity) == 1: # only intra given
                    sparsity = sparsity + [0]
                for i, sparse in enumerate(reversed(sparsity[:2])): # intra, inter
                    sparse = sparse if is_iter(sparse) else [sparse]*4
                    for j, sp in enumerate(sparse):   # e-->e, e-->i, i-->e, i-->i
                        idx = (pt.sign(self._mask_modularity) == i) * (self._mask_rec == [3, 1, -1, -3][j])
                        n = pt.sum(idx)
                        ids = pt.randperm(n)[ : int(sp * n)]
                        idc = pt.nonzero(idx)[ids].T
                        self._mask_sparsity[idc[0], idc[1]] = 0
                if len(sparsity) == 3:
                    sparse = sparsity[2] if is_iter(sparsity[2]) else [sparsity[2]]*2
                    for j, sp in enumerate(sparse):   # e-->e, e-->i, i-->e, i-->i
                        idx = pt.tile((self._mask_EI == ((1 - j) * 2 - 1))[:,None], [1, self.n_input])
                        n = pt.sum(idx)
                        ids = pt.randperm(n)[ : int(sp * n)]
                        idc = pt.nonzero(idx)[ids].T
                        self._mask_sparsity_in[idc[0], idc[1]] = 0
        else:
            if len(mask_sparsity) <= 2:  # list
                self._mask_sparsity = pt.tensor(mask_sparsity[0]).bool()
                assert self._mask_sparsity.shape == self.shape
                if len(mask_sparsity) == 2:
                    self._mask_sparsity_in = pt.tensor(mask_sparsity[1]).bool()
                    assert self._mask_sparsity_in.shape == self._Win[0].shape
                else:
                    self._mask_sparsity_in = pt.zeros_like(self._Win[0]).bool()
            else:  # single tensor
                self._mask_sparsity = pt.tensor(mask_sparsity).bool()
                assert self._mask_sparsity.shape == self.shape
                self._mask_sparsity_in = pt.zeros_like(self._Win[0]).bool()
        if not readout_feedback:
            self._mask_sparsity[:, : self.n_out] = 0
        if not direct_IO:
            self._mask_sparsity_in[: self.n_out] = 0

    def _init_weights(self, beta=None, discretize=False, uniform_weights = False):
        """
        Initialize weights from a beta distribution. Inter-/intra-module as well as
        excitatory/inhibitory connections can be parameterized individually.

        beta (list of len 0..3 or list of lists [of lists] or 0, float):
            Beta and alpha values as well as a scaling factor for drawing weights from beta distribution.
            Up to 3 lists for intra-modular and inter-modular as well the input weight matrix.
            Sublists define excitatory / inhibitory connections (e->e, e->i, i->e, i->i).
            Defaults to uniform distribution of range 0..1.
            If 0: no weight init
        discretize (bool):
            Intialize weights as int.
            Defaults to False.
        direct_IO (bool):
            If False: no input connections to readout nodes.
            Defaults to True.

        """
        if beta is None:
            beta = [[[1]*3]*4]*2 + [[[1]*3]*2]
        # initialize recurrent weights
        for i, b in enumerate(beta[:2]):   # inter, intra
            b = b if is_iter(b) else [b]*4
            for j, _b in enumerate(b): # e-->e, e-->i, i-->e, i-->i
                idx = (pt.sign(self._mask_modularity) == i) * (self._mask_rec == [3, 1, -1, -3][j])
                n = pt.sum(idx) * (1 if uniform_weights else self._bs)
                beta_gen = Beta(_b[0], _b[1])
                if discretize:
                    weights = pt.round(beta_gen.sample([n]) * (_b[2] + 1) - 0.5)
                else:
                    weights = beta_gen.sample([n]) * _b[2]
                if uniform_weights:
                    self._Wrec[:, idx] = weights[None].expand(self._bs, -1)
                else:
                    self._Wrec[idx.repeat(self._bs, 1, 1)] = weights

        self._Wrec *= self._mask_sparsity * self._mask_EI

        # initialize input weights
        if len(beta) == 3:
            b = beta[2] if len(beta[2]) == 2 else [beta[2]]*2
            for j, _b in enumerate(b): # excitatory / inhibitory (post)
                idx = (self._mask_EI == ((1 - j) * 2 - 1))[:,None].expand(-1, self.n_input)
                n = pt.sum(idx) * (1 if uniform_weights else self._bs)
                beta_gen = Beta(_b[0], _b[1])
                if discretize:
                    weights = pt.round(beta_gen.sample([n]) * (_b[2] + 1) - 0.5)
                else:
                    weights = beta_gen.sample([n]) * _b[2]
                if uniform_weights:
                    self._Win[:, idx] = weights[None].expand(self._bs, -1)
                else:
                    self._Win[idx.repeat(self._bs, 1, 1)] = weights

            self._Win *= self._mask_sparsity_in

    def reset_state(self, monitors=True):
        """ reset dynamic variables """
        self.t_last_spike[:] = pt.inf
        super().reset_state(monitors)

    def forward(self, inputs, noise=None):
        """
        Update network states.

        inputs (tensor):
            Input tensor of size (batch_size, n_input).
        noise (tensor):
            Tensor containing random noise of size (batch_size, n_reservoir).
            defaults to None: noise is computed on-the-fly

        """
        if noise is None:
            if self._noise:
                noise = self.Normal.sample((self._bs, self._N)) * self._var_noise
            else:
                noise = 0
        I_input = pt.einsum('bji,bi->bj', self._Win, inputs) + noise
        I_rec = pt.einsum('bji,bi->bj', self._Wrec, self.o)
        self.h = pt.clip(self.h + ((self.rest - self.h) + I_rec + I_input) / self._tau, *self.h_max)
        self.h[self.t_last_spike < self.t_ref] = self.reset
        # spiking
        self.o = self._f(self.h)
        self.h[self.o == 1] = self.reset
        self.t_last_spike += 1
        self.t_last_spike[self.o == 1] = 0

        self._T += 1

        for name, mon in self._monitors.items():
            if name == 'h': mon.append(self.h)
            elif name == 'o': mon.append(self.o)

        return self.o

    def run(self, inputs=None, steps=None, noise=None, stateful=True):
        """
        Update network states iteratively. Meant as a convenience function for testing.
        Depending on steps and batch size this might be faster than forward.

        inputs (tensor):
            Input tensor of size (batch_size, steps, n_input).
            Defaults to zero tensor.
        steps (int):
            Number of steps to simulate.
            Defaults to input shape if input is not None else 1.
        noise (tensor):
            Tensor containing random noise of size (batch_size, steps, n_reservoir).
            defaults to None: noise is computed on-the-fly

        """
        if steps is None:
            if inputs is not None:
                steps = inputs.shape[1]
            else:
                steps = 1
        if inputs is None:
            if self._inputs is None:
                inputs = pt.zeros((self._bs, steps, self.n_input))
            else:
                len_inp = len(self._inputs)
                inputs = self._inputs[:, self._T % len_inp : (self._T + steps) % len_inp]

        # pre-compute noise
        if noise is None:
            if self._noise:
                noise = self.Normal.sample((steps, self._bs, self._N)) * self._var_noise[None, :, :]
            else:
                noise = 0

        # pre-compute inputs
        I_input = pt.einsum('bji,bti->tbj', self._Win, inputs) + noise

        # temporary state variables
        h = pt.zeros((steps + 1, self._bs, self._N))  # 'membrane potential'
        o = pt.zeros((steps + 1, self._bs, self._N))  # binary output of spike events

        if stateful:
            h[0] = self.h
            o[0] = self.o

        # iterative update
        for t in range(steps):
            I_rec = pt.einsum('bji,bi->bj', self._Wrec, o[t])
            h[t + 1] = pt.clip(h[t] + ((self.rest - h[t]) + I_rec + I_input[t]) / self._tau, *self.h_max)
            h[t + 1][self.t_last_spike < self.t_ref] = self.reset

            # spiking
            o[t + 1] = self._f(h[t + 1])
            h[t + 1][o[t + 1] == 1] = self.reset
            self.t_last_spike += 1
            self.t_last_spike[o[t + 1] == 1] = 0

            self._T += 1

        if stateful:
            self.h = h[-1].clone()
            self.o = o[-1].clone()

        for name, mon in self._monitors.items():
            if name == 'h':  mon.extend(h[1:])
            elif name == 'o': mon.extend(o[1:])

        return o[1:].cpu().numpy(), h[1:].cpu().numpy()

    def isi(self, net_id=0, interval=[0, None], irregularity=True, filter_pause=False):
        """ compute inter-spike intervalls and irregularity score per neuron """
        spike_times = self.spike_times(net_id, interval)
        isi = []
        for _s in spike_times:
            _isi = []
            for j in range(len(_s)-1):
                i = _s[j+1] - _s[j]
                if (i < filter_pause and filter_pause > 0) or filter_pause == 0:
                    _isi.append(i)
            isi.append(_isi)

        if irregularity:
            irregularity = [np.std(_isi) / np.mean(_isi) for _isi in isi]
            return isi, np.array(irregularity)
        else:
            return isi

    def spike_times(self, net_id=0, interval=[0, None], plot=False):
        """ note spike times over some interval [_s, _e] for one network """
        _s, _e = interval
        if _e is None: _e = self._T.cpu()
        spikes = []
        for i in range(self._N):
            s = []
            for t in range(_s, _e):
                if self._monitor_o[t][net_id][i] == 1:
                    s.append(t)
            spikes.append(s)
        if plot:
            self.plot_spikes(interval, spikes)
        return spikes

    def plot_spikes(self, net_id=0, interval=[0, None], spikes=None, bin_interval=20, savedir=None):
        """ plot spikes over some time interval [_s, _e] """
        if spikes is None:
            spikes = self.spike_times(net_id, interval, plot=False)
        _s, _e = interval
        if _e is None: _e = self._T.cpu()

        # average firing rate
        b_i = bin_interval
        kernel = [1,] * (b_i + 1)
        r_mean = np.mean(self.monitor_o[_s:_e, net_id], 1)
        r_mean_smooth = np.convolve(r_mean, kernel)[b_i // 2 : -b_i // 2] / (b_i + 1)
        mpl.plot(np.arange(_s, _e), r_mean_smooth * self._N, color='black', linewidth=0.5)

        # spikes
        for i in range(self._N):
            mpl.scatter(spikes[i], np.repeat(i, len(spikes[i])), s=1, c ='black')

        # frame, axes
        mpl.xlim(_s, _e)
        mpl.ylim(0, self._N)
        for t in range(_s, _e):
            if t % bin_interval == 0:
                mpl.axvline(t, color='black', linewidth=0.5)
        mpl.xlabel('time step')
        mpl.ylabel('neuron i.d.')
        if savedir is not None:
            mpl.savefig(savedir)
        mpl.show()

#%%

class ODIN(LSM):
    """
    Simulation of 8-bit hidden state / 4-bit weight liquid state machine hardware implementation
    see Frenke, Legat, Lefebvre, & Bol (2019) https://doi.org/10.1109/TBCAS.2018.2880425

    The state of each node (neuron) is given by its binary output o and hidden state h.
        o_t+1 = step_threshold( h_t+1 )
        h_t+1 = h_t + (W_rec x o_t + W_in x inputs_t - (threshold - rest) / 2)

    h is constrained to 0..127
    weights are constrained to -7..7

    Parameters:
        batch_size (int):
            Number of networks that are evaluated in parallel for efficient GPU use.
            Network structure is identical over a batch but parameters may differ.
            Parameters must be given as basic data types or tensors of either shape
                * (1, n_reservoir)           --> node-specific parameter
                * (batch_size, 1)            --> network-specific parameter
                * (batch_size, n_reservoir)  --> node/net-specific parameter
                * (specific value [provided in doc])
        n_channels_in (int):
            Number of input _channels_, each with resolution res_input.
        n_reservoir (int):
            Number of reservoir nodes.
        n_channels_out (int):
            Number of readout _channels_, each with resolution res_readout.
            The first n (n_channels * res) neurons of the reservoir will be designated as 'readouts',
            but will still be kept in the recurrent loop by default (see readout_feedback).
            Defaults to None.
        res_input (int):
            Node resolution of the input channels. Each channel uses a population
            to encode the value of the current input state.
            Defaults to 19.
            (cf. Tang, Kumar, Yoo, & Michmizos, 2020 [CoRL])
        res_readout (int):
            Node resolution of the readout channels.
            Defaults to 8.
        k_inp (float):
            Channel-specific input scaling factor, controlling the density of input spikes.
            Defaults to uniform 1.
        sigma_inp (float):
            Spread of the Gaussian population code.
            Defaults to 1.
        p_inh (float):
            Percentage of inhibitory neurons per module.
            Defaults to 50 %.
        threshold (float or iterable or tensor[param]):
            Threshold potential for producing a spike event.
            * if iterable of length 2: set value for readout and core neurons separately
            * if tensor of shape (2, batch_size): set thresholds separately and uniquely per network
            Defaults to 7.
        h_max (tuple):
            Bounds of the hidden state.
            Defaults to (0, 127).
        w_max (tuple):
            Bounds of the weights.
            Defaults to (-7, 7).
        reset_by_subtraction (bool):
            Reset by subtracting the threshold value instead of resetting to a fixed value.
            Defaults to False (this behavior is not implemented in the ODIN hardware).
        refractory_period (int or tensor[param]):
            Period after a spike event where mebrane potential cannot be raised.
            Defaults to 3 (ms).
        n_modules (int):
            Number of internal modules of equal size.
            Defaults to None (no modularity).
        size_modules (int or list or tuple of int):
            If int: size of modules, if list/tuple: individual size of each module.
            Defaults to None (no modularity).
        sparsity (float or list of len 0..3 [of lists of len 0..4]):
            List of sparsity values for intra-, inter-modular and input connections.
            Sublists define excitatory / inhibitory connections (e->e, e->i, i->e, i->i).
            Range is [0..1] indicating the proportion of empty connections for a given type.
            Defaults to no sparsity (full connectivity).
        mask_sparsity (list of tensor[bool] of len 2):
            Pre-defined sparsity masks for recurrent and input weight matrices, respectively.
        readout_feedback (bool):
            If False: no feedback from readout nodes.
            Defaults to True.
        direct_IO (bool):
            If False: no input connections to readout nodes.
            Defaults to True.
        beta (list of len 0..3 or list of lists [of lists] or 0, float):
            Beta and alpha values as well as a scaling factor for drawing weights from beta distribution.
            Up to 3 lists for intra-modular and inter-modular as well the input weight matrix.
            Sublists define excitatory / inhibitory connections (e->e, e->i, i->e, i->i).
            Defaults to uniform distribution of range 0..1.
            If 0: no weight init
        uniform_weights (bool):
            Init all networks within the batch with the same weights.
            Defaults to False.
        monitors (str or list of string):
            Specify which variables to monitor during training.
            'h': hidden, 'o': output, 'inp': poisson_inputs
            If 'all': create all monitors.
            Defaults to None (no monitors).
        device (str):
            Device for performing tensor computations: 'cpu', 'cuda' [GPU].
            Defaults to 'cuda'.
    """
    def __init__(
            self,
            batch_size,
            n_channels_in,
            n_reservoir,
            n_channels_out = None,
            res_input = 19,
            res_readout = 8,
            k_inp = 1,
            sigma_inp = 1,
            p_inh = 0.5,
            threshold = 7,
            h_max = (0, 127),
            w_max = (-7, 7),
            reset_by_subtraction = False,
            refractory_period = 0,
            n_modules = None,
            size_modules = None,
            sparsity = None,
            mask_sparsity = None,
            readout_feedback = True,
            direct_IO = True,
            beta = None,
            uniform_weights = False,
            monitoring_variables = ('h', 'o', 'inp'),
            monitors = None,
            device = 'cuda',
        ):

        super().__init__(
                batch_size = batch_size,
                n_input = n_channels_in * res_input,
                n_reservoir = n_reservoir,
                n_channels_out = n_channels_out,
                res_readout = res_readout,
                p_inh = p_inh,
                tau = None,
                rest = 0.0,
                reset = 0.0,
                threshold = threshold,
                refractory_period = refractory_period,
                var_noise = None,
                n_modules = n_modules,
                size_modules = size_modules,
                sparsity = sparsity,
                mask_sparsity = mask_sparsity,
                readout_feedback = readout_feedback,
                direct_IO = direct_IO,
                beta = beta,
                discretize_weights = True,
                uniform_weights = uniform_weights,
                monitoring_variables = monitoring_variables,
                monitors = monitors,
                device = device,
                precision = PRECISION
            )

        self.h_max = pt.tensor(h_max)
        self.w_max = pt.tensor(w_max)

        self.n_channels_in = n_channels_in

        # input encoding parameters
        self.res_input = self.res_inp = pt.tensor(res_input)
        self.sigma_inp = pt.tensor(sigma_inp)
        self.k_inp = k_inp

        # additional spiking neuron parameters
        self.reset_by_subtraction = reset_by_subtraction

        # fix weights
        self._Wrec = pt.clip(self._Wrec, *self.w_max)
        self._Win = pt.clip(self._Win, *self.w_max)

        # constant for input generation
        self.input_scale = pt.linspace(-pt.pi / 1.5, pt.pi / 1.5, self.res_inp).repeat(self._bs, 1)

    @property
    def threshold(self):
        return self._threshold
    @threshold.setter
    def threshold(self, value):
        if is_zero_tensor(value):
            self._threshold = value
        elif is_iter(value) and len(value) == 2:  # different thresholds for readout and reservoir
            value = pt.tensor(value)
            self._threshold = pt.concat((pt.stack([value[0]] * self.n_readout),
                                        pt.stack([value[1]] * (self._N - self.n_readout))
                                        )).T
        else:
            self._threshold = pt.tensor(value)
        self._leak = pt.round(self._threshold - self.rest) // 2
    @property
    def k_inp(self):
        return self._k_inp
    @k_inp.setter
    def k_inp(self, value):
        if not hasattr(value, '__iter__'):
            value = [value]
        if is_zero_tensor(value) or len(value) == 1:
            self._k_inp = pt.repeat_interleave(pt.tensor(value), self.n_input)
        else:
            self._k_inp = pt.repeat_interleave(value, self.res_inp)
    @property
    def monitor_inp(self):
        if hasattr(self, '_monitor_inp'):
            return pt.stack(self._monitor_inp).cpu().numpy()

    def set_weights(self, weights, weight_type='rec', overwrite=True, respect_sparsity=True):
        """
        Set network weights. Only integer values allowed.

        weights (tensor):
            Weight matrices for recurrent or input weights.
            If only one matrix is given, it will be duplicated for the batch.
        weight_type (str):
            Setting either recurrent ('rec') or input ('in', 'inp') weights
        overwrite (bool):
            If False: nan-entries in 'weights' will not alter the respective weight value.
        respect_sparsity (bool):
            If True: only update weights where sparsity mask is True.

        """
        weights_no_nan = weights[~pt.isnan(weights)]
        assert pt.all(weights_no_nan >= self.w_max[0]) and pt.all(weights_no_nan <= self.w_max[1]), f"weights outside of bounds {self.w_max}"
        weights_discrete = pt.round(weights)
        assert pt.all((weights_discrete == weights)[~pt.isnan(weights)]), "only integer weights"
        super().set_weights(weights, weight_type, overwrite, respect_sparsity)

    def compute_poisson_pop(self, inputs, steps):
        """
        Compute Poisson spike trains for a Gaussian population code.

        inputs (tensor):
            Input tensor of size (batch_size, n_channels_in).
            Inputs are assumed to be scaled in the range  [-pi/2, pi/2] and
            define the center of the Gaussian population code over the interval [-pi/1.5, pi/1.5]
        steps (int):
            Length of Poisson spike trains.

        """
        p_inputs = pt.zeros((self._bs, steps+1, self.n_input))
        for i, inp in enumerate(inputs.T):
            gauss_pop = pt.exp(-0.5 * ((inp[:,None] - self.input_scale) / (self.sigma_inp * 4 / self.res_inp))**2)
            p_inputs[:, 0, i * self.res_inp : (i+1) * self.res_inp] = gauss_pop

        p_inputs *= self._k_inp
        p_inputs[:, 1:] = pt.poisson(pt.tile(p_inputs[:, :1], [1, steps, 1]))
        p_inputs[p_inputs > 1] = 1

        return p_inputs

    def forward(self, inputs=None, input_gain=1, poisson_inputs=None):
        """
        Update network states.

        inputs (tensor):
            Input tensor of size (batch_size, n_channels_in).
            May be replaced by poisson_inputs.
        input_gain (int or tensor):
            Vector controlling the gain of inputs.
            Defaults to ones.
        poisson_inputs (tensor):
            Tensor of (binary) poisson_spike trains of shape (batch_size, n_input)
            Defaults to None (will be generated on-the-fly).

        """
        if poisson_inputs is None:
            poisson_inputs = self.compute_poisson_pop(inputs, 1)[:, 1]
        poisson_inputs = input_gain * poisson_inputs
        I_input = pt.einsum('bji,bi->bj', self._Win, poisson_inputs)
        I_rec = pt.einsum('bji,bi->bj', self._Wrec, self.o)
        self.h = self.h - self._leak + I_rec + I_input
        self.h[self.h < 0] = 0
        self.h[self.h > self.h_max[1]] = 0  # overflow behaviour
        # refractory period
        if self._ref:
            self.h[self.t_last_spike < self.t_ref] = self.reset
        # spiking
        self.o = self._f(self.h)
        if self.reset_by_subtraction:
            self.h -= self.threshold * self.o
        else:
            self.h[self.o == 1] = self.reset
        if self._ref:
            self.t_last_spike += 1
            self.t_last_spike[self.o == 1] = 0

        self._T += 1

        for name, mon in self._monitors.items():
            if name == 'h': mon.append(self.h)
            elif name == 'o': mon.append(self.o)
            elif name == 'inp': mon.append(poisson_inputs)

        return self.o

    def run(self, inputs=None, steps=1, stateful=True, input_gain=1, poisson_inputs=None):
        """
        Update network states iteratively. Meant as a convenience function for testing.
        Depending on steps and batch size this might be faster than forward.

        inputs (tensor):
            Input tensor of size (batch_size, n_channels_in).
            May be replaced by poisson_inputs.
        steps (int):
            Number of steps to simulate.
            Defaults to 1.
        input_gain (int or tensor):
            Vector controlling the gain of inputs.
            Defaults to ones.
        poisson_inputs (tensor):
            Tensor of (binary) poisson_spike trains of shape (batch_size, steps, n_input)
            Defaults to None (will be generated on-the-fly).

        """
        # pre-compute inputs
        if poisson_inputs is None:
            poisson_inputs = self.compute_poisson_pop(inputs, steps)[:, 1:]
        else:
            steps = poisson_inputs.shape[1]
        poisson_inputs = input_gain * poisson_inputs

        I_input = pt.einsum('bji,bti->tbj', self._Win, poisson_inputs)

        # temporary state variables
        h = pt.zeros((steps+1, self._bs, self._N))
        o = pt.zeros((steps+1, self._bs, self._N))

        if stateful:
            h[0] = self.h
            o[0] = self.o

        # iterative update
        for t in range(steps):
            I_rec = pt.einsum('bji,bi->bj', self._Wrec, o[t])
            h[t + 1] = h[t] - self._leak + I_rec + I_input[t]
            h[t + 1][h[t + 1] < 0] = 0
            h[t + 1][h[t + 1] > self.h_max[1]] = 0  # overflow behaviour
            # refractory period
            if self._ref:
                h[t + 1][self.t_last_spike < self.t_ref] = self.reset
            # spiking
            o[t + 1] = self._f(h[t + 1])
            if self.reset_by_subtraction:
                h[t + 1] -= self.threshold * o[t + 1]
            else:
                h[t + 1][o[t + 1] == 1] =  self.reset
            if self._ref:
                self.t_last_spike += 1
                self.t_last_spike[o[t + 1] == 1] = 0

            self._T += 1

        if stateful:
            self.h = h[-1] *1
            self.o = o[-1] *1

        for name, mon in self._monitors.items():
            if   name == 'h':   mon.extend(h[1:])
            elif name == 'o':   mon.extend(o[1:])
            elif name == 'inp': mon.extend(poisson_inputs.swapaxes(0, 1))

        return o[1:].cpu().numpy(), h[1:].cpu().numpy()

#%%

class RECKON(ODIN):
    """
    Simulation of 16-bit hidden state / 8-bit weight liquid state machine hardware implementation
    see Frenkel & Indiveri (2022) https://doi.org/10.1109/ISSCC42614.2022.9731734

    The state of each node (neuron) is given by its binary output o and hidden state h.
        o_t+1 = step_threshold( h_t+1 * (tau - 1) / tau )
        h_t+1 = h_t + (W_rec x o_t + W_in x inputs_t)

    h is constrained to -32767..32767
    weights are constrained to -127..127

    Eligibility propagation learning (Bellec, Scherr, Maass, 2020) is implemented following Frenkel & Indiveri (2022)

    Parameters:
        batch_size (int):
            Number of networks that are evaluated in parallel for efficient GPU use.
            Network structure is identical over a batch but parameters may differ.
            Parameters must be given as basic data types or tensors of either shape
                * (1, n_reservoir)           --> node-specific parameter
                * (batch_size, 1)            --> network-specific parameter
                * (batch_size, n_reservoir)  --> node/net-specific parameter
                * (specific value [provided in doc])
        n_channels_in (int):
            Number of input _channels_, each with resolution res_input.
        n_reservoir (int):
            Number of reservoir nodes.
        n_channels_out (int):
            Number of readout _channels_, each with resolution res_readout.
            The first n (n_channels * res) neurons of the reservoir will be designated as 'readouts',
            but will still be kept in the recurrent loop by default (see readout_feedback).
            Defaults to None.
        res_input (int):
            Node resolution of the input channels. Each channel uses a population
            to encode the value of the current input state.
            Defaults to 19.
            (cf. Tang, Kumar, Yoo, & Michmizos, 2020 [CoRL])
        res_readout (int):
            Node resolution of the readout channels.
            Defaults to 8.
        k_inp (float):
            Channel-specific input scaling factor.
            Defaults to uniform 1.
        sigma_inp (float):
            Spread of the Gaussian population code.
            Defaults to 1.
        p_inh (float):
            Percentage of inhibitory neurons per module.
            Defaults to 50 %.
        threshold (float or iterable or tensor[param]):
            Threshold potential for producing a spike event.
            * if iterable of length 2: set value for readout and core neurons separately
            * if tensor of shape (2, batch_size): set thresholds separately and uniquely per network
            Defaults to 1024.
        h_abs_max (int):
            Maximum absolute value for the node hidden state.
            Defaults to 32767 (15 bit).
        w_abs_max (int):
            Maximum absolute value for the synpatic weights.
            Defaults to 127 (7 bit).
        stochastic_rounding (bool):
            Whether to round hidden states and weights stochstically or deterministically.
            Stochastic rounding slows down the simulation.
            Defaults to True (RECKON hardware implementation).
        reset_by_subtraction (bool):
            Reset by subtracting the threshold value instead of resetting to a fixed value.
            Defaults to True.
        refractory_period (int):
            Period after a spike event where mebrane potential cannot be raised.
            Defaults to 0 (ms) (this behavior is not implemented in the RECKON hardware).
        spiking_readout (bool):
            Whether readout nodes produce spikes or only return hidden state.
            If False: readout_feedback = False
            Defaults to True.
        learn_rate (float):
            If not None: activate learning and recording of eligibility traces.
            Defaults to None (no learning).
        freeze (triplet of bool):
            Whether to update readout, recurrent or input connections when learning.
            Defaults to False *3 (all connections learn).
        n_modules (int):
            Number of internal modules of equal size.
            Defaults to None (no modularity).
        size_modules (int or list or tuple of int):
            If int: size of modules, if list/tuple: individual size of each module.
            Defaults to None (no modularity).
        sparsity (float or list of len 0..3 [of lists of len 0..4]):
            List of sparsity values for intra-, inter-modular and input connections.
            Sublists define excitatory / inhibitory connections (e->e, e->i, i->e, i->i).
            Range is [0..1] indicating the proportion of empty connections for a given type.
            Defaults to no sparsity (full connectivity).
        mask_sparsity (list of tensor[bool] of len 2):
            Pre-defined sparsity masks for recurrent and input weight matrices, respectively.
        readout_feedback (bool):
            If False: no feedback from readout nodes.
            Defaults to True.
        direct_IO (bool):
            If False: no input connections to readout nodes.
            Defaults to True.
        beta (list of len 0..3 or list of lists [of lists] or 0, float):
            Beta and alpha values as well as a scaling factor for drawing weights from beta distribution.
            Up to 3 lists for intra-modular and inter-modular as well the input weight matrix.
            Sublists define excitatory / inhibitory connections (e->e, e->i, i->e, i->i).
            Defaults to uniform distribution of range 0..1.
            If 0: no weight init
        uniform_weights (bool):
            Init all networks within the batch with the same weights.
            Defaults to False.
        first_to_spike_input (bool):
            Whether to only use the first spike for each node in the Poisson input.
            Defaults to False.
        time_to_first_spike_coding / ttfs (bool):
            Whether to only transmit one spike per reservoir node.
            Later spikes are weighted with a negative exponential.
            Defaults to False.
        transmission_loss (float or tuple of float):
            Randomly erase a fraction of incoming spikes during simulation.
            If two values are given: set recurrent and input loss separately.
            Defaults to 0 (no loss).
        monitors (str or list of string):
            Specify which variables to monitor during training.
            'h': hidden, 'o': output, 'inp': poisson_inputs, 'tr_inp': input trace,
            'tr_rec': recurrent trace, 'tr_out': output trace, 'w', recurrent weights
            If 'all': create all monitors.
            Defaults to None (no monitors).
        device (str):
            Device for performing tensor computations: 'cpu', 'cuda' [GPU].
            Defaults to 'cuda'.
    """
    def __init__(
            self,
            batch_size,
            n_channels_in,
            n_reservoir,
            n_channels_out = None,
            res_input = 19,
            res_readout = 8,
            k_inp = 1,
            sigma_inp = 1,
            p_inh = 0.5,
            tau = 2000.,
            threshold = 1024.,
            h_abs_max = 2 ** 15 - 1,
            w_abs_max = 2 ** 7 - 1,
            stochastic_rounding = True,
            reset_by_subtraction = True,
            constant_leak = False,
            refractory_period = None,
            spiking_readout = True,
            learn_rate = None,
            freeze = [False,] * 3,
            reinforce = False,  # not implemented
            n_modules = None,
            size_modules = None,
            sparsity = None,
            mask_sparsity = None,
            readout_feedback = True,
            direct_IO = True,
            beta = None,
            uniform_weights = False,
            first_to_spike_input = False,
            time_to_first_spike_coding = False,
            ttfs = None,
            transmission_loss = False,
            monitoring_variables = ('h', 'o', 'inp', 'tr_inp', 'tr_rec', 'tr_out', 'w', 'w_in'),
            monitors = None,
            device = 'cuda',
            STDP = None,  # for compatibility reasons
            precision = PRECISION
        ):

        readout_feedback = False if not spiking_readout or learn_rate is not None else readout_feedback
        direct_IO = False if learn_rate is not None else direct_IO

        super().__init__(
                batch_size = batch_size,
                n_channels_in = n_channels_in,
                n_reservoir = n_reservoir,
                n_channels_out = n_channels_out,
                res_input = res_input,
                res_readout = res_readout,
                k_inp = k_inp,
                sigma_inp = sigma_inp,
                p_inh = p_inh,
                threshold = threshold,
                h_max = pt.tensor([-h_abs_max, h_abs_max]),
                w_max = pt.tensor([-w_abs_max, w_abs_max]),
                reset_by_subtraction = reset_by_subtraction,
                refractory_period = refractory_period,
                n_modules = n_modules,
                size_modules = size_modules,
                sparsity = sparsity,
                mask_sparsity = mask_sparsity,
                readout_feedback = readout_feedback,
                direct_IO = direct_IO,
                beta = beta,
                uniform_weights = uniform_weights,
                monitoring_variables = monitoring_variables,
                monitors = monitors,
                device = device,
            )

        # input encoding parameters
        self.first_to_spike_input = first_to_spike_input

        # spiking neuron parameters
        self.tau = tau
        self.ttfs = time_to_first_spike_coding if ttfs is None else ttfs
        if self.ttfs:
            self.t_ref = pt.inf
            self._ref = True
        self.transmission_loss = pt.tensor(transmission_loss)

        self.spiking_readout = spiking_readout

        # e-prop parameters
        self.learn_rate = None if learn_rate is None else pt.tensor(learn_rate)
        self.freeze_out, self.freeze_rec, self.freeze_inp = freeze
        if self.learn_rate is not None:
            self.trace_out = pt.zeros(self._bs, self._N)
            self.trace_rec = pt.zeros(self._bs, self._N)
            self.trace_inp = pt.zeros(self._bs, self.n_input)

        self.stochastic = stochastic_rounding
        self.constant_leak = constant_leak

    @property
    def leak(self):
        return self._leak
    
    @leak.setter
    def leak(self, value):
        self._leak = value

    @property
    def tau(self):
        return self._tau
    @tau.setter
    def tau(self, value):
        self._tau = pt.abs(pt.tensor(value))
        self._alpha = (self._tau - 1) / self._tau
        self._kappa = (pt.clip((self._tau / 10), 10, pt.inf) - 1) / pt.clip((self._tau / 10), 10, pt.inf)
    @property
    def transmission_loss(self):
        return {'rec': self._transmission_loss_rec,
                'inp': self._transmission_loss_inp}
    @transmission_loss.setter
    def transmission_loss(self, value):
        if is_iter(value) and len(value) == 2:
            self._transmission_loss_rec = value[0]
            self._transmission_loss_inp = value[1]
        else:
            self._transmission_loss_rec = value
            self._transmission_loss_inp = value
        # querying tensors in if statements is slow
        if bool(self._transmission_loss_rec) == False:
            self._transmission_loss_rec = False
        if bool(self._transmission_loss_inp) == False:
            self._transmission_loss_inp = False

    @property
    def monitor_tr(self):
        if hasattr(self, '_monitor_tr_rec'):
            return pt.stack(self._monitor_tr_rec).cpu().numpy()
    @property
    def monitor_tr_out(self):
        if hasattr(self, '_monitor_tr_out'):
            return pt.stack(self._monitor_tr_out).cpu().numpy()
    @property
    def monitor_tr_rec(self):
        if hasattr(self, '_monitor_tr_rec'):
            return pt.stack(self._monitor_tr_rec).cpu().numpy()
    @property
    def monitor_tr_inp(self):
        if hasattr(self, '_monitor_tr_inp'):
            return pt.stack(self._monitor_tr_inp).cpu().numpy()
    @property
    def monitor_w(self):
        if hasattr(self, '_monitor_w'):
            return pt.stack(self._monitor_w).cpu().numpy()
    @property
    def monitor_w_in(self):
        if hasattr(self, '_monitor_w'):
            return pt.stack(self._monitor_w_in).cpu().numpy()

    def reset_state(self, monitors=True):
        """ reset dynamic variables """
        if self.learn_rate is not None:
            self.trace_out[:] = 0
            self.trace_rec[:] = 0
            self.trace_inp[:] = 0
        super().reset_state(monitors)

    def round(self, x, stochastic=True):
        """ p(up) = fractional part, p(down) = 1 - p(up) """
        if stochastic:
            MSB = x.floor()
            LSB = x - MSB
            UP = (pt.rand_like(x) < pt.abs(LSB))
            return (MSB + pt.sign(x) * UP).to(self.precision)
        else:
            return pt.round(x)

    def compute_transmission_loss(self, connection, steps=1):
        """ Randomly set a fraction of synapses zero """
        if connection == 'rec':
            Wrec_lossy = self._Wrec.clone()
            if steps > 1:
                Wrec_lossy = Wrec_lossy[:, None].repeat(1, steps, 1, 1)
            mask = pt.rand_like(Wrec_lossy[0]) < self.transmission_loss['rec']
            return Wrec_lossy * mask
        else:
            Win_lossy = self._Win.clone()
            if steps > 1:
                Win_lossy = Win_lossy[:, None].repeat(1, steps, 1, 1)
            mask = pt.rand_like(Win_lossy[0]) < self.transmission_loss['inp']
            return Win_lossy * mask

    def compute_poisson_pop(self, inputs, steps):
        """
        Compute Poisson spike trains for a Gaussian population code.

        inputs (tensor):
            Input tensor of size (batch_size, n_channels_in).
            Inputs are assumed to be scaled in the range  [-pi/2, pi/2] and
            define the center of the Gaussian population code over the interval [-pi/1.5, pi/1.5]
        steps (int):
            Length of Poisson spike trains.

        """
        p_inputs = super().compute_poisson_pop(inputs, steps)

        # delete all spikes after the first one, per node
        if self.first_to_spike_input:
            idx = pt.argmax(p_inputs[:, 1:], 1)[:, None, :]
            i = pt.arange(steps)[None, :, None]
            mask = i <= idx
            p_inputs[:, 1:] *= mask

        return p_inputs

    def forward(self, inputs=None, input_gain=1, poisson_inputs=None):
        """
        Update network states.

        inputs (tensor):
            Input tensor of size (batch_size, n_channels_in).
            May be replaced by poisson_inputs.
        input_gain (int or tensor):
            Vector controlling the gain of inputs.
            Defaults to ones.
        poisson_inputs (tensor):
            Tensor of (binary) poisson_spike trains of shape (batch_size, n_input)
            Defaults to None (will be generated on-the-fly).

        """
        assert not self.ttfs
        assert not self.first_to_spike_input
        if poisson_inputs is None:
            poisson_inputs = self.compute_poisson_pop(inputs, 1)[:, 1]
        poisson_inputs = input_gain * poisson_inputs

        if self.transmission_loss['inp']:
            Win_lossy = self.compute_transmission_loss('inp')
            I_input = pt.einsum('bji,bi->bj', Win_lossy, poisson_inputs)
        else:
            I_input = pt.einsum('bji,bi->bj', self._Win, poisson_inputs)
        if self.transmission_loss['rec']:
            Wrec_lossy = self.compute_transmission_loss('rec')
            I_rec = pt.einsum('bji,bi->bj', Wrec_lossy, self.o)
        else:
            I_rec = pt.einsum('bji,bi->bj', self._Wrec, self.o)
        self.h = self.h + I_rec + I_input
        self.h = pt.clip(self.h, *self.h_max)
        # refractory period
        if self._ref:
            self.h[self.t_last_spike < self.t_ref] = self.reset
        # spiking
        if self.spiking_readout:
            self.o = self._f(self.h)
        else:  # apply non-linearity only to non-readout nodes
            self.o[self.n_out : ] = self._f(self.h[self.n_out : ])

        if self.reset_by_subtraction:
            self.h -= self.threshold * self.o
        else:
            self.h[self.o == 1] = self.reset

        if self.constant_leak:
            h_tmp = self.round(pt.clip(pt.abs(self.h) - self._leak, min=0), stochastic=self.stochastic)
            self.h = pt.where(self.h >= 0, h_tmp, -h_tmp)
        else:
            self.h = self.round(self._alpha * self.h, stochastic=self.stochastic)

        if self._ref:
            self.t_last_spike += 1
            self.t_last_spike[self.o == 1] = 0

        if not self.spiking_readout:  # copy hidden state to output
            self.o[ : self.n_out] = self.h[ : self.n_out]

        if self.learn_rate is not None:
            self.trace_out[:, self.n_out : ] = (self.trace_out[:, self.n_out : ] + 32 * self.o[:, self.n_out : ])
            self.trace_out = self.round(self._kappa * self.trace_out, stochastic=self.stochastic)

            self.trace_rec[:, self.n_out : ] = (self.trace_rec[:, self.n_out : ] + 32 * self.o[:, self.n_out : ])
            self.trace_rec = self.round(self._alpha * self.trace_rec, stochastic=self.stochastic)

            self.trace_inp = (self.trace_inp + 32 * poisson_inputs)
            self.trace_inp = self.round(self._alpha * self.trace_inp, stochastic=self.stochastic)

        self._T += 1

        for name, mon in self._monitors.items():
            if   name == 'h':       mon.append(self.h)
            elif name == 'o':       mon.append(self.o)
            elif name == 'inp':     mon.append(poisson_inputs)
            elif name == 'w':       mon.append(self._Wrec)
            elif name == 'w_in':    mon.append(self._Win)
            elif self.learn_rate is not None:
                if name == 'tr_out':    mon.append(self.trace_out)
                elif name == 'tr_rec':  mon.append(self.trace_rec)
                elif name == 'tr_inp':  mon.append(self.trace_inp)

        return self.o

    def run(self, inputs=None, steps=1, stateful=True, input_gain=1, poisson_inputs=None):
        """
        Update network states iteratively. Meant as a convenience function for testing.
        Depending on steps and batch size this might be faster than forward.

        inputs (tensor):
            Input tensor of size (batch_size, n_channels_in).
            May be replaced by poisson_inputs.
        steps (int):
            Number of steps to simulate.
            Defaults to 1.
        input_gain (int or tensor):
            Vector controlling the gain of inputs.
            Defaults to ones.
        poisson_inputs (tensor):
            Tensor of (binary) poisson_spike trains of shape (batch_size, steps, n_input)
            Defaults to None (will be generated on-the-fly).

        """
        # pre-compute inputs
        if poisson_inputs is None:
            poisson_inputs = self.compute_poisson_pop(inputs, steps)[:, 1:]
        else:
            steps = poisson_inputs.shape[1]
        poisson_inputs = input_gain * poisson_inputs

        if self.transmission_loss['inp']:
            Win_lossy = self.compute_transmission_loss('inp', steps)
            if steps > 1:
                I_input = pt.einsum('btji,bti->tbj', Win_lossy, poisson_inputs)
            else:
                I_input = pt.einsum('bji,bti->tbj', Win_lossy, poisson_inputs)
        else:
            I_input = pt.einsum('bji,bti->tbj', self._Win, poisson_inputs)

        if self.transmission_loss['rec']:
            Wrec_lossy = self.compute_transmission_loss('rec', steps)

        # temporary state variables
        h = pt.zeros((steps+1, self._bs, self._N))
        o = pt.zeros((steps+1, self._bs, self._N))

        if stateful:
            h[0] = self.h
            o[0] = self.o

        if self.learn_rate is not None:
            tr_out = pt.zeros((steps+1, self._bs, self._N))
            tr_out[0] = self.trace_out
            tr_rec = pt.zeros((steps+1, self._bs, self._N))
            tr_rec[0] = self.trace_rec
            tr_inp = pt.zeros((steps+1, self._bs, self.n_input))
            tr_inp[0] = self.trace_inp

        if self.ttfs:
            self.t_last_spike[:] = pt.inf  # reset refractory period
            k_t = pt.exp(-(3 * pt.arange(steps) / steps))
        else:
            k_t = pt.ones(steps)

        # iterative update
        for t in range(steps):
            if self.transmission_loss['rec']:
                if steps > 1:
                    I_rec = pt.einsum('bji,bi->bj', Wrec_lossy[:, t], o[t])
                else:
                    I_rec = pt.einsum('bji,bi->bj', Wrec_lossy, o[t])
            else:
                I_rec = pt.einsum('bji,bi->bj', self._Wrec, o[t])
            h[t + 1] = h[t] + k_t[t] * (I_rec + I_input[t])
            h[t + 1] = pt.clip(h[t + 1], *self.h_max)

            # refractory period
            if self._ref:
                self.h[self.t_last_spike < self.t_ref] = self.reset

            # spiking
            if self.spiking_readout:
                o[t + 1] = self._f(h[t + 1])
            else:  # apply non-linearity only to non-readout nodes
                o[t + 1, self.n_out : ] = self._f(h[t + 1, self.n_out : ])

            if self.reset_by_subtraction:
                h[t + 1] -= self.threshold * o[t + 1]
            else:
                h[t + 1, o[t + 1] == 1] = self.reset
            if self.constant_leak:
                h_tmp = self.round(pt.clip(pt.abs(h[t + 1]) - self._leak, min=0), stochastic=self.stochastic)
                h[t + 1] = pt.where(h[t + 1] >= 0, h_tmp, -h_tmp)
            else:
                h[t + 1] = self.round(self._alpha * h[t + 1], stochastic=self.stochastic)
             

            if self._ref:
                self.t_last_spike += 1
                self.t_last_spike[o[t + 1] == 1] = 0

            if not self.spiking_readout:  # copy hidden state to output
                o[t + 1, : self.n_out] = h[t + 1, : self.n_out]

            if self.learn_rate is not None:
                tr_out[t + 1, :, self.n_out : ] = (tr_out[t, :, self.n_out : ] + 32 * o[t + 1, :, self.n_out : ])
                tr_out[t + 1] = self.round(self._kappa * tr_out[t + 1], stochastic=self.stochastic)

                tr_rec[t + 1, :, self.n_out : ] = (tr_rec[t, :, self.n_out : ] + 32 * o[t + 1, :, self.n_out : ])
                tr_rec[t + 1] = self.round(self._alpha * tr_rec[t + 1], stochastic=self.stochastic)

                tr_inp[t + 1] = (tr_inp[t] + 32 * poisson_inputs[t])
                tr_inp[t + 1] = self.round(self._alpha * tr_inp[t + 1], stochastic=self.stochastic)

            self._T += 1

        if stateful:
            self.h = h[-1] *1
            self.o = o[-1] *1

            if self.learn_rate is not None:
                self.trace_out = tr_out[-1] *1
                self.trace_rec = tr_rec[-1] *1
                self.trace_inp = tr_inp[-1] *1

        for name, mon in self._monitors.items():
            if   name == 'h':       mon.extend(h[1:])
            elif name == 'o':       mon.extend(o[1:])
            elif name == 'inp':     mon.extend(poisson_inputs.swapaxes(0, 1))
            elif name == 'w':       mon.append(self._Wrec)
            elif name == 'w_in':    mon.append(self._Win)
            elif self.learn_rate is not None:
                if name == 'tr_out':    mon.extend(tr_out[1:])
                elif name == 'tr_rec':  mon.extend(tr_rec[1:])
                elif name == 'tr_inp':  mon.extend(tr_inp[1:])

        return o[1:].cpu().numpy(), h[1:].cpu().numpy()

    def fix_weights(self):
        """ assure correct weights """
        self._Wrec = pt.clip(self._Wrec, *self.w_max)
        self._Wrec = self.round(self._Wrec, stochastic=self.stochastic)

        self._Win = pt.clip(self._Win, *self.w_max)
        self._Win = self.round(self._Win, stochastic=self.stochastic)

    def update_weights(self, learn_signal):
        """
        Compute eligibility propagation update.
        see Frenkel & Indiveri (2022) https://doi.org/10.1109/ISSCC42614.2022.9731734

        learn_signal (tensor):
            Tensor containing loss information of size (batch_size, n_readout)
        """
        Wrec_err = self._Wrec[:, : self.n_out, :] + pt.eye(self._N)[: self.n_out, :]
        error_BP = pt.einsum('bji,bj->bi', Wrec_err.T, learn_signal)

        # recurrent / output weights
        STE_out = pt.zeros(self._bs, self._N)
        STE_out[:, : self.n_out] = 1 #- pt.tanh(self.o[:self.n_out])**2
        dWout = pt.einsum('bi,bj->bij', error_BP * STE_out, self.trace_out)

        STE_rec = pt.zeros(self._bs, self._N)
        STE_rec[(self.h > 0) * (self.h < 2 * self.threshold)] = 0.5
        STE_rec[:, : self.n_out] = 0
        dWrec = pt.einsum('bi,bj->bij', error_BP * STE_rec, self.trace_rec)

        if pt.mean(pt.abs(dWrec)) != 0:
            _d = pt.mean(pt.abs(dWout)) / pt.mean(pt.abs(dWrec))
        else:
            _d = 0

        # apply weight update
        if not self.freeze_out: self._Wrec += self.learn_rate * dWout
        if not self.freeze_rec: self._Wrec += self.learn_rate * _d * dWrec
        self._Wrec[:, :, : self.n_out] = 0

        # input weights
        dWin = pt.einsum('bi,bj->bij', error_BP * STE_rec, self.trace_inp)

        if pt.mean(pt.abs(dWin)) != 0:
            _d = pt.mean(pt.abs(dWout)) / pt.mean(pt.abs(dWin))
        else:
            _d = 0

        # apply weight update
        if not self.freeze_inp: self._Win += self.learn_rate * _d * dWin
        self._Win[:, : self.n_out, :] = 0

        self.fix_weights()