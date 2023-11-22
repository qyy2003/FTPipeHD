import torch.optim
import copy

from global_variables.config import cfg
from utils.tolist import state_dict_list,state_dict_torch

class OptimizerWithWeightVersion(torch.optim.Optimizer):
    """
        Wrapper class that holds the version of the weights
    """

    def __init__(self, model):
        self.weight_pool = {}  # 目前通过 list 来索引，pipedream 用的是 deque
        self.base_optimizer = None
        self.model = model
        self.init_optimizer(cfg)
        self.init_weight_pool()
        self.latest_version = 0
        
        # 不是每次都要调用 zero_grad
        self.batch_counter = 0
        self.update_interval = 1
    
    def init_optimizer(self, cfg):
        optimizer_cfg = copy.deepcopy(cfg.schedule.optimizer)
        name = optimizer_cfg.pop('name')
        Optimizer = getattr(torch.optim, name)
        self.base_optimizer = Optimizer(params=self.model.parameters(), **optimizer_cfg)
    
    def init_weight_pool(self):
        self.weight_pool.clear()
        self.latest_version = 0
        self.add_weight()

    def set_weights_in_weight_pool(self, version, weights):
        """
            Set the weights of the specified version in weight pool
        """
        if self.weight_pool.get(version) is None:
            print("Weights with this version {} does not exist!".format(version))
            return 
        
        self.weight_pool[version] = weights

    def add_weight(self):
        if (len(self.weight_pool) > 9):
            # print("Clearing weight pool...")
            del self.weight_pool[self.latest_version - 9]
        assert(len(self.weight_pool) <= 9)
        
        state_dict = state_dict_list(self.model.state_dict())
        store_dict = {}
        for key in state_dict:
            store_dict[key] = state_dict[key].copy()
        self.weight_pool[self.latest_version] = store_dict

    def zero_grad(self):
        if self.batch_counter > 0 and self.batch_counter % self.update_interval == 0:
            self.base_optimizer.zero_grad()

    def step(self):
        """
            Update the weight of the model with weight version update
        """
        if (self.batch_counter + 1) % self.update_interval == 0:
            # print("Perform optimizer step ... ")
            self.base_optimizer.step()
            self.latest_version += 1
            self.add_weight()
        
        self.batch_counter += 1
    

    
    def get_param_groups(self):
        return self.base_optimizer.param_groups

    def get_weight_by_version(self, version):
        return self.weight_pool[version]