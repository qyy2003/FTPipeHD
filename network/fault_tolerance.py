import requests
import orjson


def send_fault_sync(url, fail_idx, point):
    """
        Notify the worker of the fail idx and the new partition point
        Ask the worker to sync weight from replication
    """
    payload = {
        "fail_idx": fail_idx,
        "point": point
    }

    res = None
    target_url = url + '/ft/fault_sync'
    try:
        res = requests.post(target_url, orjson.dumps(payload))
        res = res.text
    except Exception as e:
        print(e)
        res = "Send fault sync fail"
    finally:
        return res


def send_restart_sync_state(url, idx, workers, points, state):
    """
        Notify the restarted worker to create new sub model and set new partiton point
    """
    target_url = url + '/ft/fault_restart_sync'
    payload = {
        "points": points,
        "idx": idx,
        "workers": workers,
        "model_args": state['model_args'],
        "model_name": state['model_name'],
        "aggr_interval": state['aggr_interval'],
        "term": state['term'],
        "profiling_interval": state['profiling_interval']
    }

    res = None
    try:
        res = requests.post(target_url, orjson.dumps(payload))
        res = res.text
    except Exception as e:
        print(e)
        res = "Fault restart sync timeout"
    finally:
        return res


def fetch_missed_weight(url, layers, fail_idx):
    target_url = url + '/ft/fetch_missed_weight'
    payload = {
        "layers": layers,
        "fail_idx": fail_idx
    }

    res = None
    try:
        res = requests.get(target_url, params=payload, timeout=10)
        res = orjson.loads(res.content)
    except Exception as e:
        print(e)
        res = "Fetch missed weight timeout"
    finally:
        return res


def commit_fault_sync(url, points, iter_id):
    """
        Notify the worker to create new sub model and set new partiton point
    """
    target_url = url + '/ft/commit_fault_sync'
    payload = {
        "points": points,
        "iter_id": iter_id
    }

    res = None
    try:
        res = requests.get(target_url, params=payload)
        res = res.text
    except Exception as e:
        print(e)
        res = "Commit fault sync timeout"
    finally:
        return res


def commit_restart_fault_sync(url, iter_id, data_len):
    """
        Notify the worker to store the store iter_id under restart condition
    """
    target_url = url + '/ft/commit_restart_fault_sync'
    payload = {
        "iter_id": iter_id,
        "data_len": data_len
    }

    res = None
    try:
        res = requests.get(target_url, params=payload)
        res = res.text
    except Exception as e:
        print(e)
        res = "Commit restart fault sync timeout"
    finally:
        return res


def send_fault_global_sync(url, stage, workers, points, iter_id, needed_params):
    """
        Send the needed params to workers for global recover
    """
    url += '/ft/fault_global_sync'
    payload = {
        'points': points,
        'params': needed_params,
        'idx': stage,
        'workers': workers,
        'iter_id': iter_id
    }

    res = None
    try:
        res = requests.post(url, orjson.dumps(payload))
        res = res.text
    except Exception as e:
        print(e)
        res = "Fault global sync timeout"
    finally:
        return res


def send_backup_weight(url, iter_id, weight):
    """
        Send weight to the next stage for replication
    """
    payload = {
        "iter_id": iter_id,
        "weight": weight
    }

    res = None
    target_url = url + '/ft/backup_weight'
    try:
        res = requests.post(target_url, orjson.dumps(payload), timeout=10)
        res = res.text
    except Exception as e:
        print(e)
        res = "Send backup weight timeout"
    finally:
        return res


def send_global_backup_weight(url, iter_id, weight):
    """
        Send weight to the central node for global replication
    """
    payload = {
        "iter_id": iter_id,
        "weight": weight
    }

    res = None
    target_url = url + '/ft/global_backup_weight'
    try:
        res = requests.post(target_url, orjson.dumps(payload), timeout=10)
        res = res.text
    except Exception as e:
        print(e)
        res = "Send global backup weight timeout"
    finally:
        return res


def fetch_global_weight_backup(url):
    """
        Fetch the weight for global backup from workers
    """
    url += '/ft/fetch_global_weight_backup'
    res = None
    try:
        res = requests.get(url)
        res = orjson.loads(res.content)
    except Exception as e:
        print(e)
        res = "fetch global weight backup timeout"
    finally:
        return res


def fetch_restart_missed_weight(url):
    target_url = url + '/ft/fetch_restart_missed_weight'
    res = None
    try:
        res = requests.get(target_url, timeout=10)
        res = orjson.loads(res.content)
    except Exception as e:
        print(e)
        res = "Fetch restart missed weight timeout"
    finally:
        return res


def send_weight_redistribute(url, failed_set, point):
    """
        Called by the central node, ask the alive workers to fetch the desired weights
    """
    payload = {
        "failed_set": failed_set,
        "point": point
    }

    res = None
    target_url = url + '/ft/weight_redistribute'
    try:
        res = requests.post(target_url, orjson.dumps(payload))
        res = res.text
    except Exception as e:
        print(e)
        res = "Send weight redistribute fail"
    finally:
        return res


def fetch_desired_weight(url, layers):
    target_url = url + '/ft/fetch_desired_weight'
    payload = {
        "layers": layers
    }

    res = None
    try:
        res = requests.get(target_url, params=payload, timeout=10)
        res = orjson.loads(res.content)
    except Exception as e:
        print(e)
        res = "Fetch missed weight timeout"
    finally:
        return res