import time
import threading
from fault_tolerance.utils import get_params_from_remote

import global_variables.common as cm_variables
import global_variables.training as tr_variables
import global_variables.fault_tolerance as ft_variables
import network.fault_tolerance as ft_network
from utils.general import get_layer_from_point, find_idx_by_layer, get_params_from_global_replication, get_params_from_local_replication, get_params_from_sub_model, load_backup_params
from utils.init import init_sub_optimizer


def sync_worker(failed_set, partition_point):
    """
        Send new worker list to the worker and ask them to sync the weight from neighbor
    """
    workers = cm_variables.get_workers()
    done = []

    def sync_thread(url, idx):
        res = ft_network.send_weight_redistribute(url, failed_set, partition_point)
        # print(res)
        if res == 'ok':
            done.append(idx)   

    for idx, url in workers.items():
        idx=int(idx)
        if idx > 0:
            threading.Thread(target=sync_thread, kwargs=dict(url=url,idx=idx)).start()

    while len(done) != len(workers) - 1:
        time.sleep(0.1)
    
    del done
    return 


def commit_workers(partition_point, iter_id):
    """
        After weight redistribution, ask all nodes to create new sub model and load new parameters
    """
    workers = cm_variables.get_workers()
    done = []

    def sync_thread(url, idx):
        res = ft_network.commit_fault_sync(url, partition_point, iter_id)
        if res == 'ok':
            done.append(idx)   
        print("Current commit ends ...")

    for idx, url in workers.items():
        idx=int(idx)
        if idx > 0:
            threading.Thread(target=sync_thread, kwargs=dict(url=url,idx=idx)).start()

    while len(done) != len(workers) - 1:
        time.sleep(0.1)
    del done


def weight_redistribute_worker_handler(failed_set, point):
    """
        Called by workers, find the desired weights and fetch them from the alive nodes
    """
    prev_point = tr_variables.get_partition_point()
    prev_idx = cm_variables.get_prev_idx()
    new_idx = cm_variables.get_stage_idx()

    old_start, old_end = get_layer_from_point(prev_point, prev_idx)
    if old_end == -1:
        old_end = tr_variables.get_total_layer() - 1

    # the layer of the previous node, fetch the local replication
    prev_start, prev_end = get_layer_from_point(prev_point, prev_idx - 1)
    if prev_end == -1:
        prev_end = tr_variables.get_total_layer() - 1

    new_start, new_end = get_layer_from_point(point, new_idx)
    if new_end == -1:
        new_end = tr_variables.get_total_layer() - 1

    # construct the M_N
    M_needed = {}
    for l in range(new_start, new_end + 1):
        if prev_idx not in failed_set and ((l >= old_start and l <= old_end) or (l >= prev_start and l <= prev_end)):
            # fetch weights locally
            if M_needed.get(prev_idx) is None:
                M_needed[prev_idx] = []
            M_needed[prev_idx].append(l)
        else:
            j = find_idx_by_layer(prev_point, l)
            if j in failed_set:
                if j + 1 == len(prev_point) + 1 or (j + 1) in failed_set:
                    if M_needed.get(0) is None:
                        M_needed[0] = []
                    M_needed[0].append(l)
                else:
                    if M_needed.get(j + 1) is None:
                        M_needed[j + 1] = []
                    M_needed[j + 1].append(l)
            else:
                if M_needed.get(j) is None:
                    M_needed[j] = []
                M_needed[j].append(l)
    
    # print("Previous point: {}, Cur point: {}".format(prev_point, point))
    # print("Failed Set: {}, worker num: {}".format(failed_set, cm_variables.get_worker_num()))
    params = {}  # the params of the new sub-model
    for idx, layers in M_needed.items():
        print("{} needs layer {} from {}".format(prev_idx, layers, idx))
        # fetch the layers from the desired node
        if idx == prev_idx:
            # fetch from local
            for l in layers:
                if l >= old_start and l <= old_end:
                    params[l] = get_params_from_sub_model(l, old_start)
                else:
                    # fetch from the local replication
                    params[l] = get_params_from_local_replication(l)
        else:
            # fetch via network 
            if idx == len(prev_point) + 1:
                idx = 0
            remote_params = get_params_from_remote(idx, layers)
            params.update(remote_params)
    
    ft_variables.store_needed_params(params)
    

def weight_redistribute_central_handler(point):
    """
        Called by the central node, find the desired weights and fetch them
    """
    prev_point = tr_variables.get_partition_point()
    cur_idx = 0
    new_idx = 0

    old_start, old_end = get_layer_from_point(prev_point, cur_idx)
    if old_end == -1:
        old_end = tr_variables.get_total_layer() - 1

    new_start, new_end = get_layer_from_point(point, new_idx)
    if new_end == -1:
        new_end = tr_variables.get_total_layer() - 1

    params = {}
    for l in range(new_start, new_end + 1):
        if l >= old_start and l <= old_end:
            # fetch weights from the local model
            params[l] = get_params_from_sub_model(l, old_start)
        else:
            params[l] = get_params_from_global_replication(l)

    return params


def commit_fault_sync_handler(points, iter_id):
    """
        Called by workers, commit the fault sync operation
    """
    params = ft_variables.get_needed_params()
    create_sub_model(points)

    load_backup_params(params, points)
    tr_variables.set_partition_point(points)
    ft_variables.rm_needed_params()

    init_sub_optimizer()

    # reset the commit message during training process
    commit = tr_variables.get_commit()
    commit['lock'].acquire()
    ft_variables.set_start_iter_id(iter_id)
    ft_variables.update_training_term()
    try:
        commit['forward_id'] = iter_id - 1
        commit['backward_id'] = iter_id - 1
        commit['lock'].notifyAll()
    finally:
        commit['lock'].release()