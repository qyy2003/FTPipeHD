import global_variables.common as cm_variables
import global_variables.training as tr_variables
import global_variables.fault_tolerance as ft_variables
from network.fault_tolerance import fetch_desired_weight
from network.offline import check_available
from utils.general import get_layer_from_point, get_params_from_global_replication, get_params_from_local_replication, get_params_from_sub_model
from utils.init import init_sub_optimizer

def find_fail_worker():
    """
        Find the failed workers
    """
    # TODO: Temporarily Flask is used, not ping as well as multi-threading transmit
    workers = cm_variables.get_workers()

    fail_idx = []
    restart_idx = [] # for reconnect workers
    for idx, url in workers.items():
        idx=int(idx)
        if idx > 0:
            # threading.Thread(fetch_each_weight, kwargs=dict(model=model, idx=idx, url=url)).start()
            res = check_available(url)
            if res == "Check available network fail" or res == "no":
                fail_idx.append(idx)
            elif not res == "Occupied":
                # worker restart
                restart_idx.append(idx)

    del workers
    return fail_idx, restart_idx


def restart_sync_state(idx, workers, partition_point, model_args, model_name, aggr_interval, term, profiling_interval):
    """
        Called by workers, sync the states under restart case
    """
    cm_variables.set_stage_idx(idx)
    cm_variables.set_model_args(model_args)
    cm_variables.set_model_name(model_name)
    cm_variables.update_workers(workers)

    ft_variables.set_training_term(term)
    
    tr_variables.update_profiling_interval(profiling_interval)
    tr_variables.set_aggregate_interval(aggr_interval)
    tr_variables.set_partition_point(partition_point)
    
    create_sub_model(partition_point)
    init_sub_optimizer()

    return 


def update_worker_by_fail_idx(fail_idx):
    """
        Update the worker list by fail idx, to make the worker list not change too much
    """
    workers = cm_variables.get_workers()
    new_worker = {
        '0': workers['0']
    }
    
    if len(fail_idx) == 1:
        for idx, url in workers.items():
            idx=int(idx)
            if idx > 0:
                if idx > fail_idx[0]:
                    new_worker[str(idx - 1)] = url
                elif idx < fail_idx[0]:
                    new_worker[str(idx)] = url
    else:
        cnt = 1
        for i in range(1, len(workers)):
            if not i in fail_idx:
                new_worker[str(cnt)] = workers[str(i)]
                cnt += 1

    print(new_worker)
    cm_variables.update_workers(new_worker)
    del workers, new_worker


def get_params_from_remote(target_idx, layers):
    """
        Get the parameters of the given layers from the target index
    """
    target_url = cm_variables.get_url_from_worker(target_idx)
    res = fetch_desired_weight(target_url, layers)
    return res['param']


def fetch_desired_weight_worker_handler(layers):
    """
        Used by the worker, return the weights of the given layers
    """
    cur_idx = cm_variables.get_stage_idx()
    point = tr_variables.get_partition_point()

    start, end = get_layer_from_point(point, cur_idx)
    if end == -1:
        end = tr_variables.get_total_layer() - 1

    params = {}
    for l in layers:
        if l >= start and l <= end:
            params[l] = get_params_from_sub_model(l, start)
        else:
            # fetch from the local replication
            params[l] = get_params_from_local_replication(l)
    
    return params


def fetch_desired_weight_central_handler(layers):
    """
        Used by the central node, return the weights of the given layers
    """
    point = tr_variables.get_partition_point()

    start, end = get_layer_from_point(point, 0)
    if end == -1:
        end = tr_variables.get_total_layer() - 1

    # used for the fetching the local replication of the last node
    last_start, last_end = get_layer_from_point(point, len(point))
    if last_end == -1:
        last_end = tr_variables.get_total_layer() - 1

    params = {}
    for l in layers:
        if l >= start and l <= end:
            params[l] = get_params_from_sub_model(l, start)
        elif l >= last_start and l <= last_end:
            # fetch from the local replication
            params[l] = get_params_from_local_replication(l)
        else:
            # fetch from the global replication
            params[l] = get_params_from_global_replication(l)
    
    return params
