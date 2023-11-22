import argparse
import torch
import copy
import threading

from global_variables.config import cfg
from global_variables.common import get_model_args, get_model_name, get_stage_idx, get_worker_num, set_semaphore, set_logger,get_device
from global_variables.training import set_all_element,set_total_layer, set_sub_model, get_sub_model, set_sub_optimizer, get_sub_optimizer, set_sub_scheduler
from global_variables.record import set_recorder

from models.general_model import init_sub_model
from utils.general import get_layer_from_point
from utils.optimizer import OptimizerWithWeightVersion
from utils.recorder import Recorder
from utils.logger import Logger


def arg_parse():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Training Partitioning Framework')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)') # 16 for yolo
    parser.add_argument('--data', type=str, default='../data/voc.yaml', help='data.yaml path')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--platform', type=str, dest='platform',
                        help='The platform you want to run on, edge or cloud, default value is edge',
                        default='edge')
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')
    parser.add_argument('--port', type=int, default=5000, help='port that run the http service')
    parser.add_argument('--mode', type=str, default='multiple', help='train on single device or multiple devices')
    parser.add_argument('--checkpoint', type=bool, default=False, help='Export the model parameter every epoch')
    parser.add_argument('--load_cp', type=bool, default=False, help='Is load the checkpoint')
    args = parser.parse_args()

    return args


def prepare_sub_model_optimizer_scheduler(point):
    """
        Create the sub-model according to the partition point
    """
    cur_stage = get_stage_idx()
    worker_num = get_worker_num()
    batch_diff=worker_num-cur_stage
    start_layer, end_layer = get_layer_from_point(point, cur_stage)
    print("Partitioned layer: {}, {}".format(start_layer, end_layer))
    set_total_layer(get_model_args()['total_layer'])

    optimizer_cfg = copy.deepcopy(cfg.schedule.optimizer)
    name = optimizer_cfg.pop('name')
    Optimizer = getattr(torch.optim, name)

    schedule_cfg = copy.deepcopy(cfg.schedule.lr_schedule)
    name = schedule_cfg.pop('name')
    Scheduler = getattr(torch.optim.lr_scheduler, name)

    device = get_device()

    for i in range(batch_diff):
        sub_model = init_sub_model(get_model_name(), get_model_args(), start_layer, end_layer)
        sub_model.to(device)
        sub_optimizer = Optimizer(params=sub_model.parameters(), **optimizer_cfg)
        sub_scheduler = Scheduler(optimizer=sub_optimizer, **schedule_cfg)
        set_all_element(sub_model, sub_optimizer, sub_scheduler)

def init_sub_optimizer():
    sub_model = get_sub_model()
    optimizer_cfg = copy.deepcopy(cfg.schedule.optimizer)
    name = optimizer_cfg.pop('name')
    Optimizer = getattr(torch.optim, name)
    sub_optimizer = Optimizer(params=sub_model.parameters(), **optimizer_cfg)
    set_sub_optimizer(sub_optimizer)
    del sub_model, sub_optimizer
# def init_sub_optimizer():
#     sub_model = get_sub_model()
#     sub_optimizer = OptimizerWithWeightVersion(sub_model)
#     set_sub_optimizer(sub_optimizer)
#     del sub_model, sub_optimizer


# init recorder for recording the loss info
def init_recorder():
    recorder = Recorder()
    set_recorder(recorder)
    del recorder

def init_sub_scheduler():
    sub_optimizer = get_sub_optimizer()
    schedule_cfg = copy.deepcopy(cfg.schedule.lr_schedule)
    name = schedule_cfg.pop('name')
    Scheduler = getattr(torch.optim.lr_scheduler, name)
    sub_scheduler = Scheduler(optimizer=sub_optimizer, **schedule_cfg)
    set_sub_scheduler(sub_scheduler)
    del sub_optimizer
# init scheduler according to config file
# def init_sub_scheduler():
#     sub_optimizer = get_sub_optimizer()
#     schedule_cfg = copy.deepcopy(cfg.schedule.lr_schedule)
#     name = schedule_cfg.pop('name')
#     Scheduler = getattr(torch.optim.lr_scheduler, name)
#     sub_scheduler = Scheduler(optimizer=sub_optimizer.base_optimizer, **schedule_cfg)
#     set_sub_scheduler(sub_scheduler)
#     del sub_optimizer


# init the semaphore according to the worker num
def init_semaphore():
    worker_num = get_worker_num()
    sem = threading.Semaphore(worker_num) 
    set_semaphore(sem)


# init the logger
def init_logger(local_rank, save_dir):
    logger = Logger(local_rank, save_dir)
    set_logger(logger)
