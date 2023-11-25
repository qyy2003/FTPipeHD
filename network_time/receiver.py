from flask import Flask, request
from transfer import MNN_to_pytorch
import orjson
import time
app = Flask(__name__)
@app.route("/network_time", methods=['POST'])
def store_labels():
    tim_st=time.time()
    payload = request.get_data()
    tim0=time.time()
    data = orjson.loads(payload)
    tim1=time.time()
    # print(" store_labels")
    # print(payload)

    idx = int(data['iter_id'])
    tim2=time.time()
    tensor1=MNN_to_pytorch(data['data'])
    tim3=time.time()

    print(idx, ": Request Data | ", tim0-tim_st)
    print(idx, ": Json Load | ", tim1 - tim0)
    print(idx, ": MNN to Tensor | ",tim3-tim2)
    print(idx, ": Total Time | ",time.time()-tim_st)
    return "ok"

if __name__=="__main__":
    # port=int(input("Input Port:"))
    port=4001
    # url = "localhost"
    # url=url+":"+str(port)
    app.run(host='0.0.0.0', port=port, threaded=True)