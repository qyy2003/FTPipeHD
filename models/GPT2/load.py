import numpy as np
import os
import sys
import copy
import torch
import random
import argparse
import torch.nn.functional as F
from tqdm import trange
import torch.nn as nn
from torch.nn.parameter import Parameter

def load_my_weight(model,state_dict, prefix_str=''):
    # 将key值进行更新weight bias
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if key.endswith(".g"):
            new_key = key[:-2] + ".weight"
            print("ggg")
        elif key.endswith(".b"):
            new_key = key[:-2] + ".bias"
            print("ggg")
        elif key.endswith(".w"):
            new_key = key[:-2] + ".weight"
            print("ggg")
        if new_key:
            print("new_key append")
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)#这行代码尝试从模型状态字典 state_dict 中获取名为 "_metadata" 的属性。这个属性通常用于存储模型状态字典的元数据，但如果不存在，它将返回 None。

    state_dict = state_dict.copy()#共享了相同的数据，因此修改副本不会影响原始对象。
    if metadata is not None:
        state_dict._metadata = metadata# 如果存在元数据，则将其设置为副本的元数据。

    def load(module, prefix=""):
        # 创建一个本地的元数据字典 local_metadata。如果没有全局元数据 metadata，则初始化为空字典；否则，它尝试从全局元数据中获取与当前前缀匹配的元数据。
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        print("local_metadata", local_metadata,"prefix",prefix)
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )
        for name, child in module._modules.items():
            
            if child is not None:
                load(child, prefix + name + ".")# 递归调用
                #print(prefix)
    start_model = model
    # 这段代码的目的是检查模型是否包含名为 "transformer" 的属性，并验证状态字典中的键是否都不以 "transformer." 开头
    if hasattr(model, "transformer") and all(not s.startswith('transformer.') for s in state_dict.keys()):
        start_model = model.transformer # 刚刚好有一个transformer属性
    load(start_model, prefix = prefix_str)

    # Make sure we are still sharing the output and input embeddings after loading weights

    return model