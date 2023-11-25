import time
import threading
import torch 
# from torchsummary import summary
import pdb

from network.offline import check_available
from global_variables.config import cfg
from global_variables.common import get_workers, get_urls, log_message, get_model_name, get_model_args, update_workers
from global_variables.training import get_aggregate_interval, get_partition_point, set_total_layer, set_partition_point
from global_variables.profiling import set_static_profiler
from models.general_model import init_model
from network.online import send_basic_info, send_workers
from utils.profiler import Profiler
from utils.scheduler import DynamicScheduler

def elect_worker():
    """
        Send election to edges to choose worker
    """
    log_message("Start electing worker...")
    workers = get_workers()
    urls = get_urls()

    def elect_thread(idx, url):
        res = check_available(url)
        # print(res)
        if res == 'ok':
            if workers.get(idx) is None:
                workers[str(idx)] = url
        print("Current election ends ...")

    for idx, url in enumerate(urls):
        idx=int(idx)
        threading.Thread(target=elect_thread, kwargs=dict(idx=idx+1, url=url)).start()

    #while len(workers) != worker_num:
    #    time.sleep(0.1)
    time.sleep(3)  # 1s 内能 ping 到几个是几个

    # print(workers)
    update_workers(workers)
    del workers, urls
    print("Total election finished ...")


def distribute_worker_set():
    """
        Send the alive worker set to all worker nodes
    """
    workers = get_workers()
    done = []
    def send_helper(idx, url):
        res = send_workers(url, idx, workers)
        if res == 'ok':
            done.append(idx)

    # self.bandwidth[0] = measure_bandwidth(workers[1]) # measure the neighbor of the master
    for idx, url in workers.items():
        idx=int(idx)
        if idx != 0:
            threading.Thread(target=send_helper, kwargs=dict(idx=idx, url=url)).start()

    while len(done) + 1 != len(workers):
        time.sleep(0.1)

    print("Distribute worker set finished ...")


def offline_profiling():
    """
        Perfrom the offline stage profiling
    """
    log_message("Start offline profiling of model {}...".format(cfg.model_name))
    ## debuging
    set_partition_point([3,10])
    # set_partition_point([11])
    return ;

    model = init_model(get_model_name(), get_model_args())
    
    # summary(model,input_size=(3,32,32),batch_size=128)
    
    # pdb.set_trace()
    inputs = torch.rand(cfg.data.batch_size, *cfg.data.input_size)
    profiler = Profiler(model, inputs)##
    profiler.static_profiling()
    profiler.bandwidth_profiling()

    set_static_profiler(profiler)
    set_total_layer(cfg.model_args.total_layer)

    dynamic_scheduler = DynamicScheduler()
    partition_point = dynamic_scheduler.calculate_partition_point(is_average=True)
    log_message("Initial Partition Point {}".format(partition_point))
    # partition_point = [9, 15]
    log_message("Actual Partition Point {}".format(partition_point))
    set_partition_point(partition_point)

    del model


def distribute_basic_info():
    """
        Send the basic information to the workers, including partition point, model name, model args...
    """
    log_message('Sending partition point and basic info to workers ... ')
    workers = get_workers()
    for idx, url in workers.items():
        idx=int(idx)
        if idx != 0:
            res = send_basic_info(url, get_partition_point(), get_model_name(), get_model_args(), get_aggregate_interval())
            assert(res == "ok")
    