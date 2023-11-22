import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy
import numpy as np
import orjson
import threading

from global_variables.common import get_model_args, get_model_name, get_workers, get_url_from_worker, get_worker_num
from network.offline import measure_neighbor_bandwidth
from global_variables.training import get_aggregate_interval, get_partition_point
from utils.general import get_layer_from_point, measure_bandwidth
import pickle

class ModelForProfiling(nn.Module):
    """
        Profiling the model in a different way of forward and backward
    """
    def __init__(self, origin_model):
        super(ModelForProfiling, self).__init__()
        self.blocks = []
        for module in origin_model.features:
            self.blocks.append(deepcopy(module))
        
        self.blocks.append(deepcopy(origin_model.classifier))
        self.forward_time = np.zeros(len(self.blocks))
        self.backward_time = np.zeros(len(self.blocks))

    def forward(self, x):
        self.output = []
        self.inputs = []
        for idx, block in enumerate(self.blocks):
            # detach from previous
            x = Variable(x.data, requires_grad=True)
            self.inputs.append(x)

            # compute output
            start_time = time.time()
            if idx == len(self.blocks) - 1:
                x = x.mean(3).mean(2)
            x = block(x)
            self.forward_time[idx] += (time.time() - start_time)

            self.output.append(x)
        return x

    # Calculate backward layer by layer
    def backward(self, g):
        for i, output in reversed(list(enumerate(self.output))):
            if i == (len(self.output) - 1):
                start_time = time.time()
                output.backward(g)
            else:
                start_time = time.time()
                output.backward(self.inputs[i + 1].grad.data)
            
            self.backward_time[i] += (time.time() - start_time)

    def get_forward_time(self):
        return self.forward_time

    def get_backward_time(self):
        return self.backward_time


class Profiler:
    """
        Wrapper of the profiling information of the offline stage
    """
    def __init__(self, origin_model, inputs):
        self.total_layers = -1
        self.module_list = []
        self.inputs = inputs
        self.model = origin_model
        self.param_size = []
        self.output_size = []
        self.mtu = 1500 # use ping to estimate the bandwidth
        self.presum_forward = []
        self.presum_backward = []
        self.bandwidth = {}
        self.computing_power = {}
        self.profiling_rounds = 1
    
    def static_profiling(self):
        """
            Profile the related info of the model, including forward, backward time of each layer, number of the layers,
            datasize of the output in each layer
        """
        self.total_layers = self.model.total_layer
        self.time_profiling()
        self.datasize_profiling()
    
    def time_profiling(self):
        print("Profiling execution time...")
        forward_time, backward_time = self.model.profile_helper(self.inputs, self.profiling_rounds)
    
        # Construct the time matrix, where presum_forward[i] denotes the execution time through layer 0 to layer i
        self.presum_backward.append(backward_time[0])
        self.presum_forward.append(forward_time[0])

        for i in range(1, self.total_layers):
            self.presum_forward.append(self.presum_forward[-1] + forward_time[i])
            self.presum_backward.append(self.presum_backward[-1] + backward_time[i])

    def datasize_profiling(self):
        print("Profiling output size of each layer...")

        def hook_fn(model, input, output):
            # count parameter size
            # TODO: 这个方法要和序列化以后的大小（网络传输的）做比较
            total_params = 0
            parameter_list = list(model.parameters())
            for p in model.parameters():
                total_params += torch.DoubleTensor([p.numel()])
            model.param_size[0] = total_params

            output_str = orjson.dumps(output.tolist())
            model.output_size[0] = len(output_str)

        for m in self.model.features:
            m.register_forward_hook(hook_fn)
            m.register_buffer('param_size', torch.zeros(1, dtype=torch.float64))
            m.register_buffer('output_size', torch.zeros(1, dtype=torch.float64))
        
        for m in self.model.classifier:
            m.register_forward_hook(hook_fn)
            m.register_buffer('param_size', torch.zeros(1, dtype=torch.float64))
            m.register_buffer('output_size', torch.zeros(1, dtype=torch.float64))

        with torch.no_grad():
            self.model(self.inputs)
        
        for m in self.model.features:
            self.param_size.append(m.param_size.item())
            self.output_size.append(m.output_size.item())
        
        for m in self.model.classifier:
            self.param_size.append(m.param_size.item())
            self.output_size.append(m.output_size.item())
    
    def bandwidth_profiling(self):
        """
            Profile the bandwidth of the device
        """
        self.bandwidth.clear()
        workers = get_workers()
        def measure(idx, url):
            # res = measure_neighbor_bandwidth(url, idx, workers, get_model_name(), get_model_args(), get_aggregate_interval())
            res = measure_neighbor_bandwidth(url)
            # res should be the bandwidth
            print(res)
            self.bandwidth[idx] = float(res)
            print("Current measure ends, idx {}...".format(idx))

        # self.bandwidth[0] = measure_bandwidth(workers[1]) # measure the neighbor of the master
        self.bandwidth[0] = 67602.88
        for idx, url in workers.items():
            idx=int(idx)
            if idx != 0:
                threading.Thread(target=measure, kwargs=dict(idx=idx, url=url)).start()

        while len(self.bandwidth) != len(workers):
            time.sleep(0.1)

        print(self.bandwidth)
        print("Bandwidth profiling finished ...")
    
    def get_time_interval(self, i, j, type):
        """
            get the execution time interval through layer i to layer j
        """
        if type == 0:
            presum_time = self.presum_forward
        elif type == 1:
            presum_time = self.presum_backward
        else:
            print("Type unknown")
            return -1

        if i > j:
            print("i should be less or equal to j!")
            return float("inf")

        if i == 0:
            return presum_time[j]
        else:
            return presum_time[j] - presum_time[i - 1]
    
    def calculate_computing_power(self, time):
        """
            Calculate the computing power of each worker from time
        """
        computing_power = {}
        local_url = get_url_from_worker(0)
        computing_power[local_url] = 1.0 # Assume computing power of master is 1
        prev_point = get_partition_point()
        # predict the computing power of each worker
        for idx, t in time.items():
            idx=int(idx)
            url = get_url_from_worker(idx)
            start_layer, end_layer = get_layer_from_point(prev_point, idx)
            if end_layer == -1:
                end_layer = self.total_layers - 1
            master_forward_time = self.get_time_interval(start_layer, end_layer, 0)
            master_backward_time = self.get_time_interval(start_layer, end_layer, 1)
            computing_power[url] = (t[0] + t[1]) / (master_forward_time + master_backward_time)

        self.computing_power = computing_power
        print("computing power:", computing_power)
        return computing_power