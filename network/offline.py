import requests
import orjson


def check_available(url):
    """
        Check whether the target worker is available for training
        res.text may be "ok" or "no", no means the worker can not train due to scarce resources
    """
    res = None
    target_url = url + '/is_available'

    try:
        res = requests.get(target_url, timeout=10)
        res = res.text
    except Exception as e:
        print(e)
        res = "Check available network fail"
    finally:
        return res


def measure_neighbor_bandwidth(url):
    """
        Ask the worker to measure the bandwidth of its next idx
    """
    res = None
    target_url = url + '/measure_neighbor'
    try:
        res = requests.get(target_url)
        res = res.content
    except Exception as e:
        print(e)
        res = "Measure neighbor timeout"
    finally:
        return res