import time
import threading

from global_variables.training import get_commit, get_partition_point, get_aggregate_interval, set_partition_point, get_profiling_interval
from global_variables.profiling import get_static_profiler
import global_variables.fault_tolerance as ft_variables
import global_variables.common as cm_variables
from utils.general import load_backup_params
from utils.init import  init_sub_optimizer
from utils.offline import distribute_worker_set
from utils.scheduler import DynamicScheduler
from fault_tolerance.utils import find_fail_worker, update_worker_by_fail_idx
from fault_tolerance.redistribution import commit_workers, sync_worker, weight_redistribute_central_handler
import network.fault_tolerance as ft_network


def backward_timeout_handler(iter_id, start_id, train_step_distribute):
    """
        This function is triggered when the timer of the backward timeout
    """
    commit = get_commit()
    if iter_id <= commit['backward_id']:
        # print("Old timeout backward result {}...".format(iter_id))
        return 
    
    if ft_variables.is_received_id(iter_id):
        # the gradients of the iter_id has been received
        ft_variables.remove_receive_id(iter_id)
        return 
    
    if ft_variables.get_fault_status() == 0 and ft_variables.get_start_iter_id() == start_id:
        # normal status and the start_id varies after recovery
        print("Backward timeout, election restart, iter_id {}!".format(iter_id))
        recover_start = time.time()

        ft_variables.set_fault_status(2)

        fault_type = fault_handler(iter_id)
        recover_end = time.time()
        print("Recover Time: {}".format(recover_end - recover_start))

        if fault_type != 0:
            # fault happens
            retrain_start = time.time()
            retrain_batch(fault_type, train_step_distribute, iter_id)
            retrain_end = time.time()
            print("Retrain Time: {}".format(retrain_end - retrain_start))

        print("Total Fault Tolerance Time: {}".format(time.time() - recover_start))

        ft_variables.set_fault_status(0)


def fault_handler(iter_id):
    """
        Determine the fault type of the fault
        return val: 0 for no faults, other for the number of the previous workers, which is used for retraining batches
    """
    # find the failed and restarted worker
    failed_set, restart_set = find_fail_worker()
    if len(restart_set) == 0 and len(failed_set) == 0:
        # No worker fails or restarts
        print("No worker fails or restarts...")
        return 0

    # Three-phase recovery
    prev_point = get_partition_point() # for calculating the needed layer after recovery
    if len(restart_set) > 0:
        # recover the restarted idx first
        print("Some workers restart immediately")
        state = {
            'model_args': cm_variables.get_model_args(),
            'model_name': cm_variables.get_model_name(),
            'aggr_interval':  get_aggregate_interval(),
            'term': ft_variables.get_training_term(),
            'profiling_interval':  get_profiling_interval()
        }

        workers = cm_variables.get_workers()
        for i in restart_set:
            cur_url = cm_variables.get_url_from_worker(i)
            res = ft_network.send_restart_sync_state(cur_url, i, workers, prev_point, state)
            assert(res == "ok")
    
    # update the worker idx
    update_worker_by_fail_idx(failed_set)
    
    # put the restart idx into the failed set
    failed_set = failed_set + restart_set

    # Model Repartitioning
    distribute_worker_set()
    partition_point = get_partition_point()
    if len(failed_set) > 0:
        static_profiler = get_static_profiler()
        static_profiler.bandwidth_profiling()
        del static_profiler
        dynamic_scheduler = DynamicScheduler()
        partition_point = dynamic_scheduler.calculate_partition_point()

    # Weight Redistribution
    sync_worker(failed_set, partition_point)

    # Get the desired params from local
    central_params = weight_redistribute_central_handler(partition_point)

    # Ask the worker to commit 
    commit_workers(partition_point, iter_id)

    # Central node create new model
    create_sub_model(partition_point)
    load_backup_params(central_params, partition_point)
    init_sub_optimizer()
    set_partition_point(partition_point)
    return len(prev_point) + 1 


def retrain_batch(prev_num, train_step_distribute, iter_id):
    """
        If fault happens, after three-phase recovery, the batch should be retrained
    """
    # re-train the batch that did not receive the backward data
    print("Re-train the batch that did not receive the backward data...")
    # First release all the semaphore
    sem = cm_variables.get_semaphore()
    for _ in range(prev_num):
        try:
            sem.release()
        except Exception as e:
            print(e)

    # reset the commit status
    commit = get_commit()
    commit['lock'].acquire()
    try:
        commit['forward_id'] = iter_id - 1
        commit['backward_id'] = iter_id - 1
        commit['lock'].notifyAll()
    finally:
        commit['lock'].release()
    
    new_sem = threading.Semaphore(cm_variables.get_worker_num())
    cm_variables.set_semaphore(new_sem)
    ft_variables.set_start_iter_id(iter_id)
    cur_batch = ft_variables.get_train_data(iter_id)
    ft_variables.update_training_term()
    recover_end = time.time()
    while not isinstance(cur_batch, int):
        new_sem.acquire()
        train_step_distribute(iter_id, cur_batch)
        iter_id += 1
        cur_batch = ft_variables.get_train_data(iter_id)
    
    print("Re-train finished")

    recover_retrain_end = time.time()
    # log_message("Fault Recovery: Recover Time: {}, retrain {}".format(recover_end - recover_start, recover_retrain_end - recover_start))