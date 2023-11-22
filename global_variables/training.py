import threading
import torch.nn as nn
import global_variables
import torch
from global_variables.common import get_device

total_layer = -1
partition_point = None

sub_model = []
sub_optimizer = []
sub_scheduler = []
sub_input= []
sub_output = []
# state_dict={}


commit = dict(forward_id=-1, backward_id=-1, lock=threading.Condition(), epoch=-1, data_len=-1, train_info={})

time_profiling_interval = 10

labels_pool = {}  # for worker storing the labels
inter_pool = {} # storing intermediate result
id_weight_version = {} # storing the weight version used by the batch 

criterion = nn.CrossEntropyLoss()

aggregate_interval = -1


def set_total_layer(layer_):
    global total_layer
    total_layer = layer_


def get_total_layer():
    global total_layer
    return total_layer


def get_partition_point():
    global partition_point
    return partition_point


def set_partition_point(point_):
    global partition_point
    partition_point = point_

def get_all_backward_element():
    global sub_model,sub_optimizer,sub_scheduler,sub_input,sub_output
    return sub_model[0],sub_optimizer[0],sub_scheduler[0],sub_input.pop(0),sub_output.pop(0)

def get_all_forward_element():
    global sub_model,sub_optimizer,sub_scheduler
    return sub_model[0],sub_optimizer[0],sub_scheduler[0]

def save_all_training_element(sub_model_,sub_optimizer_,sub_scheduler_,sub_input_,sub_output_):
    global sub_model,sub_optimizer,sub_scheduler,sub_input,sub_output
    sub_model.append(sub_model_)
    sub_optimizer.append(sub_optimizer_)
    sub_scheduler.append(sub_scheduler_)
    sub_input.append(sub_input_)
    sub_output.append(sub_output_)
    sub_model.pop(0)
    sub_optimizer.pop(0)
    sub_scheduler.pop(0)

def set_all_element(sub_model_, sub_optimizer_, sub_scheduler_):
    global sub_model, sub_optimizer, sub_scheduler
    sub_model.append(sub_model_)
    sub_optimizer.append(sub_optimizer_)
    sub_scheduler.append(sub_scheduler_)

def get_optimizer_lr():
    return sub_optimizer[0].param_groups[0]['lr']

def set_optimizer_lr(lr):
    global sub_optimizer
    for sub_optimizer_ in sub_optimizer:
        for param_group in sub_optimizer_.param_groups:
            param_group['lr'] = float(lr)

def set_sub_model(model_):
    global sub_model
    sub_model.append(model_)


def get_sub_model():
    global sub_model
    return sub_model


def get_sub_optimizer():
    global sub_optimizer
    return sub_optimizer[0]


def set_sub_optimizer(optimizer_):
    global sub_optimizer
    sub_optimizer.append(optimizer_)


def get_commit():
    global commit
    return commit


def get_sub_scheduler():
    global sub_scheduler
    return sub_scheduler


def set_sub_scheduler(scheduler_):
    global sub_scheduler
    sub_scheduler = scheduler_


def set_commit_len(data_len):
    global commit
    commit['data_len'] = data_len


# reset the weight pool for each epoch
def reset_weight_pool():
    sub_optimizer.init_weight_pool()


def reset_batch_counter():
    global sub_optimizer
    sub_optimizer.batch_counter = 0


def update_profiling_interval(interval_):
    global time_profiling_interval
    time_profiling_interval = interval_


def get_profiling_interval():
    global time_profiling_interval
    return time_profiling_interval


def set_train_mode():
    global sub_model
    for sub_model_ in sub_model:
        sub_model_.train()


def store_inter_result(iter_id, data, type):
    """
        Store the intermediate result for backward
        type: 0 for input, 1 for output
    """
    global inter_pool
    if inter_pool.get(iter_id) is None:
        inter_pool[iter_id] = [0] * 2
    
    inter_pool[iter_id][type] = data


def get_inter_result(iter_id, type):
    global inter_pool
    return inter_pool[iter_id][type]


def remove_inter_result(iter_id):
    global inter_pool
    del inter_pool[iter_id]


def update_labels_pool(iter_id, label):
    global labels_pool
    labels_pool[iter_id] = label


def delete_label(iter_id):
    global labels_pool
    del labels_pool[iter_id]


def get_label(iter_id):
    global labels_pool
    return labels_pool[iter_id]


def set_aggregate_interval(aggregate_interval_):
    global aggregate_interval
    aggregate_interval = aggregate_interval_


def get_aggregate_interval():
    global aggregate_interval
    return aggregate_interval


def get_weight_version(iter_id):
    global id_weight_version
    assert(id_weight_version.get(iter_id) is not None)
    return id_weight_version[iter_id]


def set_weight_version(iter_id, version):
    global id_weight_version
    id_weight_version[iter_id] = version


def reset_weight_version():
    global id_weight_version
    id_weight_version.clear()

# def get_state_dict():
#     global state_dict
#     model = sub_model[0]
#     state_dict = model.state_dict()
def weight_aggregate():
    # print("start")
    cur_stage = global_variables.common.get_stage_idx()
    worker_num = global_variables.common.get_worker_num()
    batch_diff = worker_num - cur_stage
    global sub_model#, state_dict
    model=sub_model[0]
    state_dict = model.state_dict()
    for key in state_dict:
        state_dict[key] = torch.zeros(state_dict[key].shape).to(get_device())
    for model_ in sub_model:
        state_dict_ = model_.state_dict()
        for key in state_dict:
            state_dict[key]+= state_dict_[key]
            # if(((state_dict_[key]-state_dict[key])*(state_dict_[key]-state_dict[key])).sum()>0.01):
            #     print(key)
            #     print(((state_dict_[key]-state_dict[key])*(state_dict_[key]-state_dict[key])).sum())
    for key in state_dict:
        state_dict[key] /=batch_diff
    for model_ in sub_model:
        model_.load_state_dict(state_dict)
