import inspect
import tinyodin as to
import numpy as np
import torch as pt
import json
from .model import TinyOdinModel

def instanciate_model(source_path, input_spike_interval, batch_size=1, device="cpu", n_cpu_cores=None):
    state_dict = pt.load(source_path, map_location=pt.device(device))
    # Filter sd to only include keys that match MyModel's __init__ parameters
    model_init_params = inspect.signature(TinyOdinModel.__init__).parameters
    filtered_sd = {k: v for k, v in state_dict.items() if k in model_init_params}
    filtered_sd["input_spike_interval"] = input_spike_interval

    m = TinyOdinModel(batch_size, device, n_cpu_cores=n_cpu_cores, **filtered_sd)
    m.load_state_dict(state_dict, strict=False)
    return m

def read_entire_synapse_memory(sim):
    n_neurons = 256
    vals = np.arange(n_neurons * n_neurons) 
    for pre, post in [(x, y) for x in range(n_neurons) for y in range(n_neurons)]:
        vals[pre*n_neurons + post] = sim.read_single_synapse_memory(pre, post)
    w = decode_synapse_memory(vals)
    return w.reshape(n_neurons, n_neurons)

def twos_complement_to_decimal(x, n_bits):
    mask = (1 << n_bits) - 1
    sign = 1 << (n_bits - 1)
    
    y = (x & mask)
    if not isinstance(y, int):
        y = y.astype(int)
    
    y = y - ((y & sign) << 1)

    return y

def decimal_to_twos_complement(x, n_bits):
    if isinstance(x, int):
        return x if x >= 0 else (2 << (n_bits -1)) + x
    else: 
        return np.where(x >= 0, x, x + (2 << (n_bits -1)))

def cartesian_product_s(s):
    x = np.repeat(np.arange(s, dtype=np.uint16), s)
    y = np.tile(np.arange(s, dtype=np.uint16), s)
    return x, y

def decode_neuron_memory(val):
    # 1 bit disable, 7 bit leak_str, 12 thr, 12 core
    disable = (val >> 31) & 0x1
    leak_str = (val >> 24) & 0x7F
    thr = (val >> 12) & 0xFFF
    core = val & 0xFFF
    thr = twos_complement_to_decimal(thr, 12)
    core = twos_complement_to_decimal(core, 12)
    return disable, leak_str, thr, core 

def decode_synapse_memory(val):
    # 1 bit sign, 3 bit weight
    sign = (val >> 3) & 0x1
    weight = val & 0x7
    return (-2 * sign + 1) * weight

def print_decode_n(val):
    disable, leak_str, thr, core = decode_neuron_memory(val)
    print("Read neuron memory:", ["disable:", disable, "leak_str:", leak_str, "thr:", thr, "core:", core])

def print_decode_s(val):
    weight = decode_synapse_memory(val)
    print("Read synapse memory:", weight)


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
    # reset to stop it from spiking
    sim.read_output_spikes() # consume any spikes generated before the reset
    # set configs
    sim.set_config(to.SpiConfigRegisterIndex.SPI_OPEN_LOOP, spi_open_loop)
    sim.set_config(to.SpiConfigRegisterIndex.SPI_AER_SRC_CTRL_nNEUR, spi_aer_src_ctrl_nneur)
    sim.set_config(to.SpiConfigRegisterIndex.SPI_MAX_NEUR, spi_max_neur)
    return sim

def init_sim_with_params(leak_str, thr, weights, disable=np.full(256, False, dtype=bool), spi_open_loop=False, spi_aer_src_ctrl_nneur=False, spi_max_neur=255):
    leak_str = np.full((256,), leak_str)
    thr = np.full((256,), thr)
    # if spi_aer_src_ctrl_nneur is True, it gets events from scheduler -> outputs input spike events as well
    sim = init_sim(spi_open_loop=spi_open_loop, spi_aer_src_ctrl_nneur=spi_aer_src_ctrl_nneur, spi_max_neur=spi_max_neur) 
    neuron_idx = np.arange(256)
    sim.write_neuron_memory(neuron_idx, disable, leak_str, thr)
    pre = np.repeat(neuron_idx, 256)
    post = np.tile(neuron_idx, 256)
    weights = weights.flatten()
    s = (weights < 0)
    w = np.abs(weights)
    sim.write_synapse_memory(pre, post, s, w)
    sim.reset()
    sim.read_output_spikes()
    return sim


def reset_sim(sim):
    sim.set_rst(0)
    sim.cycle_clk(25)
    sim.set_rst(1)
    sim.cycle_clk(25)
    sim.set_rst(0)
    sim.cycle_clk(25)

def send_neuron_event(sim, neuron_idx):
    payload = (neuron_idx & 0xFF)
    sim.send_input(to.AerInEventType.AER_IN_NEURON_EVENT, payload)

def send_time_reference_event(sim, n_times=1):
    payload = int("11111111", 2)
    for i in range(0, n_times):
        sim.send_input(to.AerInEventType.AER_IN_TREF_EVENT, payload)

def send_virtual_event(sim, neuron_idx, sign, weight):
    opcode = "10"
    payload = ((sign & 0x1) << 5) + ((weight & 0x7) << 4) + (neuron_idx & 0xF)
    print(payload)
    data = (int(opcode, 2) << 8) + payload 
    print(data)
    sim.send_input(data, 1)

# load agent data from file
def unpack_agent_info(agent_key, data, info: dict):
    params = dict(info["simple"])
    for name in info["arrays"]:
        params[name] = data[f"{agent_key}__{name}"]
    return params

def agent_id_from_key(agent_key: str):
    a, b, c = map(int, agent_key.split("_"))
    return (a, b, c)

def load_agents_npz(path: str):
    out = {}
    with np.load(path, allow_pickle=True) as data:
        meta = json.loads(data["__meta__"].item())
        for agent_key, info in meta.items():
            agent_id = agent_id_from_key(agent_key)            
            out[agent_id] = unpack_agent_info(agent_key, data, info)
    return out