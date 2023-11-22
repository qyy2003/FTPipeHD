import threading
import time

from global_variables.fault_tolerance import set_fault_status, update_training_term, set_start_iter_id, get_train_data, store_needed_params, get_needed_params
from global_variables.common import get_workers, get_stage_idx, get_worker_num, get_url_from_worker, get_semaphore
from global_variables.training import get_partition_point, get_total_layer, get_sub_model, set_partition_point, get_commit
from network.dynamic_scheduler import send_update_partition_point, fetch_ds_missed_weight, commit_weight_sync
from utils.general import get_layer_from_point, find_idx_by_layer, load_backup_params
from utils.init import  init_sub_optimizer
from utils.tolist import state_dict_list

def ds_get_missing_layer(prev_point, point):
    """
        Get missing layer given the current point and the updated point
    """
    cur_stage = get_stage_idx()
    prev_start_layer, prev_end_layer = get_layer_from_point(prev_point, cur_stage)
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
        if needed_stage.get(target_stage) is None:
            needed_stage[target_stage] = []
        needed_stage[target_stage].append(i)
    
    return needed_stage, origin_layer


def ds_get_needed_params(needed_stage):
    """
        Fetch the parameters from other workers according to the given layers in the dynamic scheduling
    """
    needed_parameters = {} # storing the needed param from other workers
    print("needed_stage ", needed_stage)
    for idx, layers in needed_stage.items():
        if idx != get_stage_idx():
            target_url = get_url_from_worker(idx)
            res = fetch_ds_missed_weight(target_url, layers)
            needed_parameters.update(res['param'])
            del res
    return needed_parameters


def ds_get_needed_origin_params(origin_layer, prev_point):
    """
        Get the parameters from origin sub model by origin layer in dynamic_scheduling
    """
    cur_stage = get_stage_idx()

    start_layer, _ = get_layer_from_point(prev_point, cur_stage)
    
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


def dynamic_scheduling(point, iter_id, train_step_distribute):
    """
        Dynamically update the partition point
    """
    set_fault_status(3)
    workers = get_workers()
    done = []

    def sync_thread(url, idx):
        res = send_update_partition_point(url, point, iter_id)
        # print(res)
        if res == 'ok':
            done.append(idx)   

    for idx, url in workers.items():
        idx=int(idx)
        if idx > 0:
            threading.Thread(target=sync_thread, kwargs=dict(url=url,idx=idx)).start()
    
    # Weight sync of the master
    prev_point = get_partition_point()
    needed_stage, origin_layer = ds_get_missing_layer(prev_point, point)
    needed_params = ds_get_needed_params(needed_stage)
    needed_origin_params = ds_get_needed_origin_params(origin_layer, prev_point)

    # create and load the backup parameters
    needed_params.update(needed_origin_params)
    done.append(0)
    while len(done) != len(workers):
        time.sleep(0.1)
        
    # commit partition point update
    workers = get_workers()
    done.clear()

    def sync_thread(url, idx):
        res = commit_weight_sync(url, point, iter_id)
        if res == 'ok':
            done.append(idx)   
        print("Current {} commit weight sync ends ...".format(idx))

    for idx, url in workers.items():
        idx=int(idx)
        if idx > 0:
            threading.Thread(target=sync_thread, kwargs=dict(url=url,idx=idx)).start()
        
    while len(done) != len(workers) - 1:
        time.sleep(0.1)
    
    # update local parameters
    create_sub_model(point)
    
    load_local_params(needed_params, point)
    init_sub_optimizer()
    set_partition_point(point)
    del done

    print("Re-train the batch after the newly created sub model...")
    # First release all the semaphore
    sem = get_semaphore()
    for i in range(get_worker_num()):
        try:
            sem.release()
        except Exception as e:
            print(e)

    # reset the commit status
    commit = get_commit()
    commit['lock'].acquire()
    try:
        commit['forward_id'] = iter_id
        commit['backward_id'] = iter_id
        commit['lock'].notifyAll()
    finally:
        commit['lock'].release()

    iter_id += 1
    set_start_iter_id(iter_id)
    update_training_term()
    cur_batch = get_train_data(iter_id)
    while not isinstance(cur_batch, int):
        sem.acquire()
        print("{} retrain start".format(iter_id))
        train_step_distribute(iter_id, cur_batch)
        iter_id += 1
        cur_batch = get_train_data(iter_id)
    
    set_fault_status(0)


def load_local_params(needed_params: dict, point: list):
    """
        Load the backup parameters after partitioning new sub model
    """
    sub_model = get_sub_model()
    start_layer, _ = get_layer_from_point(point, get_stage_idx())

    for layer, params in needed_params.items():
        if layer < sub_model.origin_features_len:
            sub_model.features[layer - start_layer].load_state_dict(params)
        else:
            if start_layer >= sub_model.origin_features_len:
                sub_model.classifier[layer - start_layer].load_state_dict(params)
            else:
                sub_model.classifier[layer - sub_model.origin_features_len].load_state_dict(params)


def update_partition_point_handler(points):
    set_fault_status(3)
    commit = get_commit()
    commit['lock'].acquire()
    try:
        commit['lock'].notifyAll()
    finally:
        commit['lock'].release()
    
    prev_point = get_partition_point()

    needed_stage, origin_layer = ds_get_missing_layer(prev_point, points)
    print("needed stage ", needed_stage)
    print("needed origin layer ", origin_layer)

    needed_params = ds_get_needed_params(needed_stage)
    needed_origin_params = ds_get_needed_origin_params(origin_layer, prev_point)

    # store the needed parameters and wait for the commit signal from master
    needed_params.update(needed_origin_params)
    store_needed_params(needed_params)


def fetch_ds_missed_weight_handler(layers):
    cur_stage = get_stage_idx()

    prev_point = get_partition_point()
    start_layer, end_layer = get_layer_from_point(prev_point, cur_stage)
    if end_layer == -1:
        end_layer = get_total_layer() - 1

    parameters = {}  # key is the layer, value is the corresponding parameters
    sub_model = get_sub_model()

    for layer in layers:
        if layer == end_layer and end_layer == get_total_layer():
            parameters[layer] = state_dict_list(sub_model.classifier.state_dict())
        else:
            parameters[layer] = state_dict_list(sub_model.features[layer - start_layer].state_dict()) # load parameters
    
    return parameters

def commit_weight_sync_handler(points, iter_id):
    needed_params = get_needed_params()
    create_sub_model(points)

    load_backup_params(needed_params, points)
    set_partition_point(points)
    del needed_params

    init_sub_optimizer()

    # reset the commit message during training process
    commit = get_commit()
    commit['lock'].acquire()
    set_start_iter_id(iter_id + 1)
    update_training_term()
    set_fault_status(0)
    try:
        commit['forward_id'] = iter_id
        commit['backward_id'] = iter_id
        commit['lock'].notifyAll()
    finally:
        commit['lock'].release()
    