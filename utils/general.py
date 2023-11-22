import os
import copy

from torch import sub
import torch

from global_variables.common import get_workers, get_worker_num, get_stage_idx, get_url_from_worker, get_model_args, get_model_name, log_message
from global_variables.training import get_partition_point, get_sub_model, get_sub_optimizer, get_total_layer
from global_variables.fault_tolerance import get_global_weight_backup, get_weight_backup
from network.online import fetch_weight
from network.fault_tolerance import fetch_missed_weight
from models.general_model import init_sub_model
from ping3 import ping

from utils.tolist import state_dict_list,state_dict_torch

def rename_key(key: str, start):
    """
        Rename the key to conduct weight sync
        The index in sub model is different from that in the whole model
    """
    new_key = ""
    if 'features' in key:
        keys = key.split('.')
        keys[1] = str(int(keys[1]) + start)
    
        for e in keys:
            new_key = new_key + e + "."
        
        new_key = new_key[:-1]
    else:
        new_key = key
    return new_key


def rank_filter(func):
    def func_filter(local_rank=-1, *args, **kwargs):
        if local_rank < 1:
            return func(*args, **kwargs)
        else:
            pass

    return func_filter


@rank_filter
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def weight_sync():
    """
        Synchronize the weight of all the models from worker
    """
    workers = get_workers()
    partition_point = get_partition_point()

    # local sync, note that local weight need not to rename the key
    worker_num = get_worker_num()
    sub_models = [0] * worker_num
    sub_models[0] = get_sub_model()

    for idx, url in workers.items():
        idx=int(idx)
        if idx > 0:
            res = fetch_weight(url)
            weight = res['weight']
            cur_start, cur_end = get_layer_from_point(partition_point, idx)
            cur_sub_model = init_sub_model(get_model_name(), get_model_args(), cur_start, cur_end)
            cur_sub_model.load_state_dict(state_dict_torch(weight))
            sub_models[idx] = cur_sub_model
    
    # model.load_state_dict(cur_weight)
    return sub_models


def get_keys_from_origin(model, cur_start, cur_end):
    """
        Get key names from the model according to the start layer and end layer
    """
    keys = []
    features_len = len(model.features)
    print("get_keys_from_origin(): layer {}, {}".format(cur_start, cur_end))
    for i in range(cur_start, cur_end + 1):
        if i < features_len:
            keys.append(model.features[i].state_dict().keys())
        else:
            keys.append(model.classifier[i - features_len].state_dict().keys())
    
    return keys


def weight_aggregation_bk(idx):
    """
        Oct.11 2022 updated
        Perform the weight aggregation using the latest 3 version in weight pool as well as updating these aggregated weights
        Note that the weight aggregation is needed as long as the vertical sync is used
    """
    sub_optimizer = get_sub_optimizer()
    sub_model = get_sub_model()
    aggregate_num = get_worker_num()

    if len(sub_optimizer.weight_pool) < aggregate_num:
        print("Weight pool length is less than 3!")
        return 

    cur_version = sub_optimizer.latest_version
    target_weights = sub_optimizer.weight_pool[cur_version]

    average_weight = {}
    for key in target_weights:
        average_weight[key] = target_weights[key].clone()
        for i in range(1, aggregate_num):
            average_weight[key] += sub_optimizer.weight_pool[cur_version - i][key].clone()
    
    for key in average_weight:
        average_weight[key] /= aggregate_num

    for i in range(1, aggregate_num):
        sub_optimizer.set_weights_in_weight_pool(cur_version - i, average_weight)
    
    sub_model.load_state_dict(average_weight)
    sub_optimizer.latest_version += 1
    sub_optimizer.add_weight()


def weight_aggregation(idx):
    """
        Perform the weight aggregation using the latest #device version in weight pool
        Note that vertical sync is not used here
    """
    sub_optimizer = get_sub_optimizer()
    sub_model = get_sub_model()
    stage_num = get_worker_num()
    aggregate_num = stage_num - idx # without vertical sync
    if aggregate_num == 1:
        # The last stage does not need to aggregate
        print("\033[36mNo aggregation is needed\033[0m")
        sub_optimizer.latest_version += 1
        sub_optimizer.add_weight()
        return 

    if len(sub_optimizer.weight_pool) < stage_num:
        print("\033[36mWeight pool length is less than {}!\033[0m".format(stage_num))
        return 

    print("\033[36mStart weight aggregation with the latest {} weights...\033[0m".format(aggregate_num))
    cur_version = sub_optimizer.latest_version
    target_weights = sub_optimizer.weight_pool[cur_version]

    average_weight = state_dict_torch(target_weights)
    for key in target_weights:
        for i in range(1, aggregate_num):
            average_weight[key] = average_weight[key]+state_dict_torch(sub_optimizer.weight_pool[cur_version - i])[key]
        average_weight[key] = torch.div(average_weight[key], aggregate_num)
    
    sub_model.load_state_dict(average_weight)
    sub_optimizer.latest_version += 1
    sub_optimizer.add_weight()


def weight_aggregation_vs(idx):
    """
        Perform the weight aggregation using the latest #device version in weight pool
        Note that vertical sync is considered here
    """
    sub_optimizer = get_sub_optimizer()
    sub_model = get_sub_model()
    stage_num = get_worker_num()
    # aggregate_num = stage_num - idx # without vertical sync
    aggregate_num = stage_num # with vertical sync
    # if aggregate_num == 1:
    #     # The last stage does not need to aggregate
    #     sub_optimizer.latest_version += 1
    #     sub_optimizer.add_weight()
    #     return 

    if len(sub_optimizer.weight_pool) < stage_num:
        print("\033[36mWeight pool length is less than {}!\033[0m".format(stage_num))
        return 

    print("\033[36mStart weight aggregation ...\033[0m")
    cur_version = sub_optimizer.latest_version
    target_weights = sub_optimizer.weight_pool[cur_version]

    average_weight = {}
    for key in target_weights:
        average_weight[key] = target_weights[key].clone()
        for i in range(1, aggregate_num):
            average_weight[key] += sub_optimizer.weight_pool[cur_version - i][key].clone()
    
    for key in average_weight:
        average_weight[key] /= aggregate_num

    sub_model.load_state_dict(average_weight)
    sub_optimizer.latest_version += 1
    sub_optimizer.add_weight()


def get_layer_from_point(point, stage_idx):
    """
        Get the start layer and end layer from partition point
    """
    worker_num = len(point) + 1   # worker_num can be deduced from the length of the point
    if stage_idx == 0:
        return 0, point[0]
    elif stage_idx == worker_num - 1:
        return point[stage_idx - 1] + 1, -1
    else:
        print(point)
        print(stage_idx)
        return point[stage_idx - 1] + 1, point[stage_idx]


def find_idx_by_layer(point, layer):
    for idx in range(len(point)):
        if layer <= point[idx]:
            return idx
    
    return len(point)# The last stage


def get_missing_layer(prev_point, point, fail_idx):
    """
        Get the missing layer and its corresponding stage by previous partition point and current point
    """
    cur_stage = get_stage_idx()
    prev_stage = cur_stage
    if cur_stage >= fail_idx:
        prev_stage += 1
    prev_start_layer, prev_end_layer = get_layer_from_point(prev_point, prev_stage)
    if prev_end_layer == -1:
        prev_end_layer = get_total_layer() - 1

    start_layer, end_layer = get_layer_from_point(point, cur_stage)
    if end_layer == -1:
        end_layer = get_total_layer() - 1

    needed_stage = {} # key is the new stage idx, value is the lists of layer
    needed_layer = []
    if start_layer > prev_end_layer or end_layer < prev_start_layer:
        # no intersection
        for i in range(start_layer, end_layer + 1):
            needed_layer.append(i)
    else:
        if start_layer < prev_start_layer:
            # layer from the previous stage is needed
            for i in range(start_layer, prev_start_layer):
                needed_layer.append(i)
        if end_layer > prev_end_layer:
            # layer from the next stage is needed
            for i in range(prev_end_layer + 1, end_layer + 1):
                needed_layer.append(i)
    
    # Find the layer needed from the origin model
    origin_layer = []
    for i in range(start_layer, end_layer + 1):
        if i not in needed_layer:
            origin_layer.append(i)

    # target stage indicates the stage which holds the parameters after rescheduling
    for i in needed_layer:
        target_stage = find_idx_by_layer(prev_point, i)
        if target_stage > fail_idx:
            # after recovery, idx decrease by 1
            target_stage -= 1
        elif target_stage == fail_idx and fail_idx == len(prev_point):
            # The fail stage is the final stage, the parameters stores in the stage 0
            target_stage = 0
        
        if needed_stage.get(target_stage) is None:
            needed_stage[target_stage] = []
        needed_stage[target_stage].append(i)
    
    return needed_stage, origin_layer


def get_needed_params(needed_stage, fail_idx, origin_layer):
    """
        Fetch the parameters from other workers backup according to the given layers
    """
    needed_parameters = {} # storing the needed param from other workers
    print("needed_stage ", needed_stage)
    for idx, layers in needed_stage.items():
        if idx != get_stage_idx():
            target_url = get_url_from_worker(idx)
            res = fetch_missed_weight(target_url, layers, fail_idx)
            needed_parameters.update(res['param'])
            del res
        else:
            # load from local backup
            local_param = fetch_missed_weight_from_local(layers, origin_layer)
            needed_parameters.update(local_param)
    return needed_parameters


def get_needed_origin_params(origin_layer, prev_point, fail_idx):
    """
        Get the parameters from origin sub model by origin layer
    """
    cur_stage = get_stage_idx()
    prev_stage = cur_stage
    if prev_stage >= fail_idx:
        prev_stage += 1

    start_layer, _ = get_layer_from_point(prev_point, prev_stage)
    
    needed_origin_params = {} 
    sub_model = get_sub_model()
    for layer in origin_layer:
        if layer < sub_model.origin_features_len:
            needed_origin_params[layer] = state_dict_list(sub_model.features[layer - start_layer].state_dict())
        else:
            if start_layer >= sub_model.origin_features_len:
                needed_origin_params[layer] = state_dict_list(sub_model.classifier[layer - start_layer].state_dict())
            else:
                needed_origin_params[layer] = state_dict_list(sub_model.classifier[layer - sub_model.origin_features_len].state_dict())

    return needed_origin_params


def fetch_missed_weight_from_local(layers, origin_layer):
    """
        Fetch the parameters from the local backup according to the given layers
    """
    weight_backup = get_weight_backup()['weight']
    local_param = {}
    for layer in layers:
        if layer not in origin_layer:
            local_param[layer] = weight_backup[layer]

    return local_param


def load_backup_params(needed_params: dict, point: list):
    """
        Load the backup parameters after partitioning new sub model
    """
    sub_model = get_sub_model()
    start_layer, _ = get_layer_from_point(point, get_stage_idx())
    for layer, params in needed_params.items():
        layer=int(layer)
        params=state_dict_torch(params)
        if layer < sub_model.origin_features_len:
            sub_model.features[layer - start_layer].load_state_dict(params)
        else:
            if start_layer >= sub_model.origin_features_len:
                sub_model.classifier[layer - start_layer].load_state_dict(params)
            else:
                sub_model.classifier[layer - sub_model.origin_features_len].load_state_dict(params)


def global_redistribute_params(prev_point, point):
    """
        Redistribute the parameters according to the new partition point
        Return the global backup parameters indexed by stage idx
    """
    new_worker_num = get_worker_num()
    needed_params = {}
    for idx in range(new_worker_num):
        layer_in_stage = {}
        start_layer, end_layer = get_layer_from_point(point, idx)
        if end_layer == -1:
            end_layer = get_total_layer() - 1
        
        for layer in range(start_layer, end_layer + 1):
            # find the idx which contains the layer
            temp_stage = find_idx_by_layer(prev_point, layer)
            if layer_in_stage.get(temp_stage) is None:
                layer_in_stage[temp_stage] = []
            
            layer_in_stage[temp_stage].append(layer)

        needed_params[idx] = layer_in_stage
    print("global_redistribute_params(): Needed params: ", needed_params)
    return needed_params


def isPointEqual(partition_point, cur_point):
    """
        Check whether two partition point equals
    """
    for i in range(len(partition_point)):
        if partition_point[i] != cur_point[i]:
            return False
    return True


def measure_bandwidth(url):
    """
        Measure the bandwidth between current edge and the target url
    """
    size = 736
    total_size = 1500 # send 1500 bytes and receive 1500 bytes
    total_time = 0
    for i in range(11):
        time_elapse = ping(url.split(':')[1][2:], size=size)
        if time_elapse is not None:
            # print("It took {} second".format(time_elapse))
            total_time += time_elapse
    if total_time > 0:
        print("It took {} second on average, the bandwidth is {}".format(total_time / 10, total_size / (total_time / 10)))
        return total_size / (total_time / 10)

    print("Can not measure the bandwidth")
    return -1


def load_checkpoint():
    """
        Load the checkpoint of the model weights
    """
    log_message("Loading the checkpoint weights")
    stage_idx = get_stage_idx()
    epoch = 0
    path = "./model_state/sub_model_{}_epoch_{}.pkl".format(stage_idx, epoch)
    sub_model = get_sub_model()
    sub_model.load_state_dict(torch.load(path))


def get_params_from_sub_model(l, start):
    """
        Given the layer l, return its corresponding parameters in sub-model
        start layer is used to distinguish the features and classifier
    """
    sub_model = get_sub_model()
    if l < sub_model.origin_features_len:
        return state_dict_list(sub_model.features[l - start].state_dict())

    if start >= sub_model.origin_features_len:
        return state_dict_list(sub_model.classifier[l - start].state_dict())
    
    return state_dict_list(sub_model.classifier[l- sub_model.origin_features_len].state_dict())


def get_params_from_local_replication(l):
    """
        Given the layer l, return its corresponding parameters in local replication
    """
    weights_lr = get_weight_backup()['weight']
    # print(weights_lr)
    assert(weights_lr.get(str(l)) is not None)
    # return state_dict_torch(weights_lr[str(l)])
    return weights_lr[str(l)]


def get_params_from_global_replication(l):
    """
        Given the layer l, return its corresponding parameters in global replication
    """
    weights_lr = get_global_weight_backup()
    assert(weights_lr.get(str(l)) is not None)

    # return state_dict_torch(weights_lr[str(l)])
    return weights_lr[str(l)]