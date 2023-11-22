import orjson
from flask import request, Blueprint

import global_variables.training as train_variables
import global_variables.fault_tolerance as ft_variables

from fault_tolerance.utils import fetch_desired_weight_worker_handler, restart_sync_state
from fault_tolerance.redistribution import weight_redistribute_worker_handler, commit_fault_sync_handler

from utils.tolist import state_dict_list

ft_route = Blueprint('ft', __name__)

@ft_route.route('/fetch_weight', methods=['POST'])
def fetch_weight():
    sub_model = train_variables.get_sub_model()
    res = dict()
    res['weight'] = state_dict_list(sub_model.state_dict())
    res['error_code'] = 0

    return orjson.dumps(res)


@ft_route.route('/backup_weight', methods=['POST'])
def backup_weight():
    payload = request.get_data()
    req = orjson.loads(payload)
    ft_variables.set_weight_backup(int(req['iter_id']), req['weight'])
    return "ok"


@ft_route.route('/fault_restart_sync', methods=['POST'])
def fault_restart_sync():
    """
        store the basic state variables needed for training
    """
    payload = request.get_data()
    req = orjson.loads(payload)

    stage_idx = req['idx']
    workers = req['workers']
    partition_point = req['points']
    model_args = req['model_args']
    model_name = req['model_name']
    aggr_interval = req['aggr_interval']
    # term = req['term']
    profiling_interval = req['profiling_interval']

    restart_sync_state(stage_idx, workers, partition_point, model_args, model_name, aggr_interval, term, profiling_interval)
    return "ok"


@ft_route.route('/commit_fault_sync', methods=['GET'])
def commit_new_model():
    # create new sub model and set new partitioning point
    points = request.args.getlist('points', type=int)
    iter_id = request.args.get('iter_id', type=int) # the next training batch id
    commit_fault_sync_handler(points, iter_id)
    
    return "ok"


@ft_route.route('/weight_redistribute', methods=['POST'])
def weight_redistribute():
    payload = request.get_data()
    req = orjson.loads(payload)
    failed_set = req['failed_set']
    point = req['point']

    weight_redistribute_worker_handler(failed_set, point)
    return "ok"


@ft_route.route('/fetch_desired_weight', methods=['GET'])
def fetch_desired_weight():
    layers = request.args.getlist('layers', type=int)
    
    res = {}
    res['param'] = fetch_desired_weight_worker_handler(layers)
    return orjson.dumps(res)

