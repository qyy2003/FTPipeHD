import torch
from transfer import pytorch_to_MNN
import requests
import orjson
import random
import time
idx=0


def send(url,batch_size,seq_len,hidden_size):
    global idx
    idx+=1
    tim_st=time.time()
    tensor0=torch.randn([batch_size,seq_len,hidden_size]);
    tim0=time.time()
    mnn0=pytorch_to_MNN(tensor0)
    print(idx, ": Tensor to MNN | ", time.time() - tim0)
    target_url = url + "/network_time"
    payload = {
        "iter_id": idx,
        "data": mnn0
    }
    tim0 = time.time()
    payload=orjson.dumps(payload)
    print(idx, ": Json Dumps | ", time.time() - tim0)
    try:
        res = requests.post(target_url, payload, timeout=10)
        res = res.text
    except Exception as e:
        print(e)
        res = "Send labels timeout"
    finally:
        print(idx, ": Total Time | ", time.time() - tim_st)
        return res

if __name__=="__main__":
    # port=int(input("Input Port:"))
    port=4001
    url = "localhost"
    url="http://"+url+":"+str(port)
    for hidden_size in [768,4096,1024]:
        for seq_len in [256,1024]:
            for bts in [1,2,4,8]:
                send(url,bts,seq_len,hidden_size)