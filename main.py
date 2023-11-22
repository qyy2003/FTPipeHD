from global_variables.config import cfg, load_config
from global_variables.common import set_program_args, set_model_args, set_model_name,set_device
from global_variables.training import set_aggregate_interval
from utils.init import arg_parse, init_logger
from utils.offline import distribute_basic_info, distribute_worker_set, elect_worker, offline_profiling
from utils.train_central import start_train
from utils.train_single import train_single

from flask_api.central.index import app

import torch
import numpy as np
import threading
import time
import logging
def main():
    # global device
    # get_summary()
    # 读取命令parameter并解析
    args = arg_parse()
    # Seed Initialization
    init_seed = 1
    torch.manual_seed(init_seed)
    np.random.seed(init_seed) # 用于numpy的随机数

    # cuda related
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device:",device)
    set_device(device)
    #读取config文件并获取相关参数
    load_config(cfg, args.config)
    set_model_name(cfg.model_name)
    set_model_args(cfg.model_args)
    set_aggregate_interval(cfg.weight_aggregation_interval)
    set_program_args(args)

    if args.mode == "single":
        single_train(args)
    else:
        collaborative_train(args)


def single_train(args):
    init_logger(args.local_rank, cfg.save_dir + "_master_single")
    train_single()

    
def collaborative_train(args):
    init_logger(args.local_rank, cfg.save_dir + "_master")

    # offline stage
    elect_worker()
    distribute_worker_set()
    offline_profiling()##
    distribute_basic_info()
    # log = logging.getLogger('werkzeug')
    # log.setLevel(logging.ERROR)
    threading.Thread(target=start_train).start()
    app.run(host='0.0.0.0', port=args.port, threaded=True)


if __name__ == '__main__':
    main()