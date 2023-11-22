import requests
import orjson


def send_update_partition_point(url, point, iter_id):
    """
        Update the partition point in the dynamic scheduling
    """
    url += '/ds/update_partition_point'
    res = None
    payload = {
        'points': point,
        'iter_id': iter_id
    }

    try:
        res = requests.get(url, params=payload)
        res = res.text
    except Exception as e:
        print(e)
        res = "Update_partition_point timeout"
    finally:
        return res


def fetch_ds_missed_weight(url, layers):
    target_url = url + '/ds/fetch_ds_missed_weight'
    payload = {
        "layers": layers,
    }

    res = None
    try:
        res = requests.get(target_url, params=payload, timeout=10)
        res = orjson.loads(res.content)
    except Exception as e:
        print(e)
        res = "Fetch ds missed weight timeout"
    finally:
        return res


def commit_weight_sync(url, points, iter_id):
    """
        Notify the worker to create new sub model and set new partiton point in dynamic scheduling
    """
    target_url = url + '/ds/commit_weight_sync'
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
        res = "Commit weight sync timeout"
    finally:
        return res