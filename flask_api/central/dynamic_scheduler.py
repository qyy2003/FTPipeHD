import orjson
from flask import request, Blueprint

from utils.dynamic_scheduler import fetch_ds_missed_weight_handler

ds_route = Blueprint('ds', __name__)

@ds_route.route('/fetch_ds_missed_weight', methods=['GET'])
def return_ds_missed_weight():
    layers = request.args.getlist('layers', type=int)

    parameters = fetch_ds_missed_weight_handler(layers)
    res = {}
    res['param'] = parameters
    return orjson.dumps(res)