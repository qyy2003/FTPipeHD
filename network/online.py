import requests
import orjson
import time
import torch
from flask_api.transfer import pytorch_to_MNN
# def send_partition_point(url, partition_point):
#     """
#         Send the partition point to the worker
#     """
#     payload = {
#         "point": partition_point
#     }
#
#     res = None
#     target_url = url + '/partition'
#     try:
#         res = requests.post(target_url, params=payload, timeout=10)
#         res = res.text
#     except Exception as e:
#         print(e)
#         res = "Send partition points timeout"
#     finally:
#         return res


def send_start_epoch(url, epoch, lr, data_len):
    """
        Notify the worker of the start of the epoch
    """
    payload = {
        "epoch": epoch,
        "lr": lr,
        "len": data_len
    }

    res = None
    target_url = url + '/start_epoch'
    try:
        res = requests.get(target_url, params=payload, timeout=10)
        res = res.text
    except Exception as e:
        print(e)
        res = "Send start epoch timeout"
    finally:
        return res
    

def fetch_weight(url):
    """
        Fetch the weight according to the model_idx
    """
    url += '/ft/fetch_weight'
    res = None
    try:
        res = requests.post(url, timeout=10)
        res = orjson.loads(res.content)
    except Exception as e:
        print(e)
        res = "Fetch weight timeout"
    finally:
        return res
    

def send_labels(url, iter_id, labels):
    """
        Send labels to the target server which calculate the loss
    """
    payload = {
        "iter_id": iter_id,
        "labels": pytorch_to_MNN(labels)
    }
    # print(payload)
    res = None
    target_url = url + "/labels"
    try:
        res = requests.post(target_url, orjson.dumps(payload), timeout=10)
        res = res.text
    except Exception as e:
        print(e)
        res = "Send labels timeout"
    finally:
        return res


# Send the intermediate output of the data to another edge
def send_train_forward(url, iter_id, data, idx, version, term=None, lr=None):
    time0=time.time();
    # print(data['hidden_states'].dtype)
    data0=pytorch_to_MNN(data)
    # print(data)
    payload = {
        "iter_id": iter_id,
        "data": data0,
        "model_idx": idx,
        "version": version,
        # "term": term,
    }
    time1 = time.time();
    # print("tolist:",str(time1-time0))
    orjson.dumps(payload)
    time0 = time.time();
    # print("orjson.dump:",str(time0 - time1))
    if lr is not None:
        payload["lr"] = lr

    res = None
    target_url = url + '/handle_forward'
    #print(payload['data'][0])
    # print("Transmitted data len: ", len(orjson.dumps(payload['data'])))
    try:
        time0 = time.time();
        res = requests.post(target_url, orjson.dumps(payload), timeout=100)
        # TODO:
        time1 = time.time();
        # print("send_train_forward:",str(time1-time0))
        res = res.text
    except Exception as e:
        print(e)
        res = "Send train forward timeout"
    finally:
        return res


def send_train_backward(url, ret):
    res = None
    traget_url = url + '/send_train_backward'
    try:
        # print(ret)
        res = requests.post(traget_url, orjson.dumps(ret), timeout=20)
        res = res.text
    except Exception as e:
        print(e)
        res = "Send train backward timeout"
    finally:
        return res


def send_workers(url, idx, workers):
    """
        Send the alive worker set
    """
    payload = {
        "idx": idx,
        "workers": workers
    }

    res = None
    traget_url = url + '/update_workers'
    print("SEND:")
    print(orjson.dumps(payload))
    try:
        res = requests.post(traget_url, orjson.dumps(payload), timeout=10)
        res = res.text
    except Exception as e:
        print(e)
        res = "Send worker timeout"
    finally:
        return res


def send_basic_info(url, point, model_name, model_args, aggr_interval):
    payload = {
        "point": point,
        "model_name": model_name,
        "model_args": model_args,
        "aggr_interval": aggr_interval
    }

    res = None
    traget_url = url + '/set_basic_info'
    try:
        res = requests.post(traget_url, orjson.dumps(payload), timeout=100)
        res = res.text
    except Exception as e:
        print(e)
        res = "Send_basic_info timeout"
    finally:
        return res