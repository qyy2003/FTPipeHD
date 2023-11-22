import requests
import orjson
from global_variables.common import get_workers
def post_data(url):
    """
        Send the partition point to the worker
    """
    payload = {
        "idx": 4,
        "workers": get_workers()
    }
    pl=str(orjson.dumps(payload),encoding="utf-8");
    print(str(orjson.dumps(payload),encoding="utf-8"))
    res = None
    target_url = url + '/updateWorkers'
    try:
        res = requests.post(target_url,pl, timeout=10)
        res = res.text
    except Exception as e:
        print(e)
        res = "Send partition points timeout"
    finally:
        return res


if __name__ == '__main__':
    # urls="http://10.192.66.75:5002"
    urls="http://192.168.50.176:50000"
    post_data(urls)