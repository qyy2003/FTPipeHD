import orjson
import requests
import threading
import torch
from flask import Flask, request
from test_data import MNNtenor
import time


app = Flask(__name__)
@app.route('/handleForward', methods=['POST'])
def get_data():
    payload = request.get_data()
    with open("json.txt", 'wb') as f:
        f.write(payload)
    # with open("json.txt", 'r') as f:
    #     payload=f.read()
    # payload=payload[2:-1]
    # print(payload[:10])
    req = orjson.loads(payload)
    # print(req)
    params={};
    params['lr']=req['lr'];
    params['iterId']=req['iterId'];
    params['modelIdx']=req['modelIdx'];
    params['version']=req['version'];
    data=[];
    for i in range(0,len(req['data']),3):
        data.append(MNNtenor(req['data'][i:i+3]))
    post_data("http://10.42.0.62:50000",params,data)
    return "ok"

def post_data(url,params,datas):
    """
        Send the partition point to the worker
    """
    data=[]
    for item in datas:
        data.extend(item.export())
    payload = {
        'data':data,
        'lr':params['lr'],
        'iterId':params['iterId'],
        'modelIdx':params['modelIdx'],
        'version':params['version']
    }
    with open("json_output.txt", 'wb') as f:
        f.write(orjson.dumps(payload,option=orjson.OPT_SERIALIZE_NUMPY ))
    # pl=str(orjson.dumps(payload),encoding="utf-8");
    # print(str(orjson.dumps(payload),encoding="utf-8"))
    res = None
    target_url = url + '/handleForward'
    try:
        res = requests.post(target_url,orjson.dumps(payload,option=orjson.OPT_SERIALIZE_NUMPY ), timeout=10)
        res = res.text
    except Exception as e:
        print(e)
        res = "Send partition points timeout"
    finally:
        return res


if __name__ == '__main__':
    # urls="http://10.192.66.75:5002"
    urls="http://10.192.133.137:5002"
    # get_data();
    # threading.Thread(target=partition).start()
    # app.run(host='0.0.0.0', port=5102, threaded=True)
    app.run(host='10.42.0.1', port=5102, threaded=True)
