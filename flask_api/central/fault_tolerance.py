import orjson
from flask import request, Blueprint
from fault_tolerance.utils import fetch_desired_weight_central_handler

from global_variables.fault_tolerance import set_weight_backup, get_weight_backup, set_global_weight_backup


ft_route = Blueprint('ft', __name__)

@ft_route.route('/backup_weight', methods=['POST'])
def backup_weight():
    payload = request.get_data()
    req = orjson.loads(payload)
    set_weight_backup(int(req['iter_id']), req['weight'])
    return "ok"


@ft_route.route('/global_backup_weight', methods=['POST'])
def global_backup_weight():
    payload = request.get_data()
    req = orjson.loads(payload)
    set_global_weight_backup(req['weight'])
    return "ok"


@ft_route.route('/fetch_desired_weight', methods=['GET'])
def fetch_desired_weight():
    layers = request.args.getlist('layers', type=int)
    
    res = {}
    res['param'] = fetch_desired_weight_central_handler(layers)
    return orjson.dumps(res)