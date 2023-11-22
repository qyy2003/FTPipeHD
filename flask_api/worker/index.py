import orjson
import threading
import torch
from flask import Flask, request
import time
from utils.train_worker import handle_forward_intermediate, handle_backward_intermediate, init_epoch, measure_neighbor_handler,  set_basic_info_handler
from global_variables.training import update_labels_pool
from global_variables.common import get_stage_idx, update_workers, set_stage_idx,get_device

from flask_api.worker.fault_tolerance import ft_route
from flask_api.worker.dynamic_scheduler import ds_route


app = Flask(__name__)
app.register_blueprint(ft_route, url_prefix="/ft")
app.register_blueprint(ds_route, url_prefix="/ds")

from flask_api.transfer import MNN_to_pytorch
@app.route('/handle_forward', methods=['POST'])
def index():
    # print("Active thread num {}".format(threading.active_count()))
    payload = request.get_data()
    time0 = time.time();
    req = orjson.loads(payload)
    time1 = time.time();
    req["data"]=MNN_to_pytorch(req["data"]);
    time0 = time.time();
    # print("to_tensor:", str(time0 - time1))
    threading.Thread(target=handle_forward_intermediate, kwargs=dict(req=req)).start()
    del req
    return "ok"


@app.route('/send_train_backward', methods=['POST'])
def receive_train_backward():
    payload = request.get_data()
    # print("receive_train_backward")
    # print(payload)
    req = orjson.loads(payload)
    # print(req)
    threading.Thread(target=handle_backward_intermediate, kwargs=dict(req=req)).start()
    return "ok"


@app.route('/labels', methods=['POST'])
def store_labels():
    payload = request.get_data()
    data = orjson.loads(payload)
    # print(" store_labels")
    # print(payload)

    iter_id = data['iter_id']
    # testlabel=torch.tensor(MNN_to_pytorch(data['labels'])[0])
    update_labels_pool(iter_id, MNN_to_pytorch(data['labels'])[0].to(dtype=torch.long))

    return "ok"


@app.route('/start_epoch', methods=['GET'])
def start_epochs():
    epoch_id = request.args.get('epoch')
    lr = request.args.get('lr')
    data_len = request.args.get('len')
    init_epoch(epoch_id, lr, data_len)
    return "ok"


@app.route('/measure_neighbor', methods=['GET'])
def measure_neighbor():
    bw = measure_neighbor_handler()
    return str(bw)


# @app.route('/partition', methods=['POST'])
# def partition():
#     # partition the model according the partitioning point
#     point = request.args.getlist('point', type=int)
#     partition_handler(point)
#
#     return "ok"


@app.route('/is_available', methods=['GET'])
def is_available():
    # Resources utilization can be checked here
    # If not resources is available, "no" can be return
    if get_stage_idx() != -1:
        return "Occupied"
    return "ok"


@app.route('/update_workers', methods=['POST'])
def update_workers_by_set():
    payload = request.get_data()
    data = orjson.loads(payload)
    idx = int(data['idx'])
    workers = data['workers']
    # print("WORKDERS:")
    # print(workers)
    set_stage_idx(idx)
    update_workers(workers)
    return "ok"


@app.route('/set_basic_info', methods=['POST'])
def set_basic_info():
    payload = request.get_data()
    data = orjson.loads(payload)
    print("set_basic_info")
    print(data)
    set_basic_info_handler(data)
    return "ok"