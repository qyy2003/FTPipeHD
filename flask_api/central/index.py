from flask import Flask, request
import orjson
import threading

from utils.train_central import handle_backward_intermediate

from flask_api.central.fault_tolerance import ft_route
from flask_api.central.dynamic_scheduler import ds_route

app = Flask(__name__)
app.register_blueprint(ft_route, url_prefix="/ft")
app.register_blueprint(ds_route, url_prefix="/ds")

@app.route('/send_train_backward', methods=['POST'])
def receive_train_backward():
    payload = request.get_data()
    req = orjson.loads(payload)
    # print("received!!")
    # print(req)
    threading.Thread(target=handle_backward_intermediate, kwargs=dict(req=req)).start()
    return "ok"