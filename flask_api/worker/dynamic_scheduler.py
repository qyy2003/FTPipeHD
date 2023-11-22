import orjson
from flask import request, Blueprint

from utils.dynamic_scheduler import update_partition_point_handler, fetch_ds_missed_weight_handler, commit_weight_sync_handler

ds_route = Blueprint('ds', __name__)

@ds_route.route('/update_partition_point', methods=['GET'])
def update_point():
    points = request.args.getlist('points', type=int)
    update_partition_point_handler(points)
   
    return "ok"


@ds_route.route('/fetch_ds_missed_weight', methods=['GET'])
def return_ds_missed_weight():
    layers = request.args.getlist('layers', type=int)

    parameters = fetch_ds_missed_weight_handler(layers)
    res = {}
    res['param'] = parameters
    return orjson.dumps(res)


@ds_route.route('/commit_weight_sync', methods=['GET'])
def commit_weight_sync():
    # create new sub model and set new partitioning point
    points = request.args.getlist('points', type=int)
    iter_id = request.args.get('iter_id', type=int) # the next training batch id
    commit_weight_sync_handler(points, iter_id)

    return "ok"