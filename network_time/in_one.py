import torch
from transfer import pytorch_to_MNN,MNN_to_pytorch
import requests
import orjson
import random
import time
idx=0


def send(batch_size,seq_len,hidden_size):
    global idx
    idx+=1
    tim_st=time.time()
    tensor0=torch.randn([batch_size,seq_len,hidden_size]);
    tim0=time.time()
    mnn0=pytorch_to_MNN(tensor0)
    payload = {
        "iter_id": idx,
        "data": mnn0
    }
    payload=orjson.dumps(payload)
    tim1 = time.time()
    data = orjson.loads(payload)
    idx = int(data['iter_id'])
    tensor1 = MNN_to_pytorch(data['data'])
    tim2 = time.time()
    return (tim1-tim0),(tim2-tim1)

if __name__=="__main__":
    # port=int(input("Input Port:"))
    tim=[[],[]]
    for hidden_size,seq_len in [[768,256],[1024,256],[4096,256]]:
        tim0=[[],[]]
        for bts in [1,8]:
            sum0=0
            sum1=0
            for i in range(10):
                x1,x2=send(bts,seq_len,hidden_size)
                sum0+=x1
                sum1+=x2
            sum0=round(sum0*100,2)
            sum1=round(sum1*100,2)
            tim0[0].append(sum0)
            tim0[1].append(sum1)
        tim[0].append(tim0[0])
        tim[1].append(tim0[1])
    # tim = list(map(list, zip(*tim)))
    print(tim)