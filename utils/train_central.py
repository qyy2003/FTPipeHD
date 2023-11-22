import orjson
import profile
import time
import threading
import torch
from memory_profiler import profile
from copy import deepcopy

import global_variables.training as train_variables
import global_variables.fault_tolerance as ft_variables
from global_variables.profiling import get_static_profiler
from global_variables.common import get_workers, is_load_cp, log_message, set_stage_idx, get_urls, get_semaphore, get_worker_num, get_stage_idx, get_url_from_worker,get_device
from global_variables.config import cfg
from global_variables.record import reset_time, set_time_per_batch, get_time_per_batch, get_recorder

from utils.init import prepare_sub_model_optimizer_scheduler, init_semaphore, init_sub_optimizer, init_recorder, init_sub_scheduler
from network.online import send_start_epoch, send_labels, send_train_forward
from utils.test import test_distribute
from utils.general import load_checkpoint, weight_aggregation, isPointEqual
from fault_tolerance.replication import replicate_weight
from fault_tolerance.handler import backward_timeout_handler
from utils.scheduler import DynamicScheduler
from utils.dynamic_scheduler import dynamic_scheduling
from utils.visualize import write_computing_graph
from custom_datasets.general_init import init_dataset
from tqdm.auto import tqdm
import copy
from flask_api.transfer import MNN_to_pytorch

id_now=0
correct, total = 0, 0
train_loss, counter = 0, 0


# progress_bar = tqdm(range(len(train_dataloader)))
progress_bar = 0
lock=threading.Condition()

def start_train():
    log_message("Training mode set to Collaborative Mode ...")
    
    partition_point = train_variables.get_partition_point()
    log_message('Creating model...')

    set_stage_idx(0)
    prepare_sub_model_optimizer_scheduler(partition_point)
    # train_variables.get_state_dict()
    # if is_load_cp():
    #     load_checkpoint()
    # check the computing graph
    # inputs = torch.rand(1, 3, 28, 28)
    # write_computing_graph(inputs, train_variables.get_sub_model())

    log_message('Setting up dataset {}...'.format(cfg.data.name))
    train_dataloader, test_dataloader = init_dataset(cfg.data.name)

    # init_recorder()
    init_semaphore()

    train_distribute(train_dataloader, test_dataloader)
    #train_single(model, logger, train_dataloader, test_dataloader)


def train_distribute(train_dataloader, test_dataloader):
    """
        Train the dataset in distribute way
    """

    log_message('Start formal training...')
    total_train_time = 0
    for epoch in range(1, cfg.schedule.total_epochs + 1):
        global progress_bar
        progress_bar = tqdm(range(len(train_dataloader)-1))
        start_time = time.time()
        lr = train_variables.get_optimizer_lr()
        # sub_optimizer.param_groups[0]['lr']
        for idx, url in get_workers().items():
            idx=int(idx)
            if idx > 0:
                send_start_epoch(url, epoch, lr, len(train_dataloader))

        # train_variables.reset_batch_counter()

        reset_time()
        global  id_now
        id_now = 0
        acc,train_loss=train_epoch_distribute(cfg, train_dataloader)

        log_msg = 'epoch {}| loss: {:.4f},acc: {:.4f}%'.format(epoch, train_loss, acc)
        log_message(log_msg)

        end_time = time.time()
        total_train_time += (end_time - start_time)
        log_message("Distribute Train phase | Epoch {}, Train time {} seconds, Total time {} minutes".format(epoch, end_time - start_time, total_train_time / 60))



def train_epoch_distribute(cfg, train_dataloader):
    """
        Perform the training of every epoch
    """
    global correct, total,train_loss, counter,progress_bar
    # print("START YES")
    for iter_id, batch in enumerate(train_dataloader):
        train_variables.set_train_mode()
        # print("YES1:",iter_id)
        sem=get_semaphore()
        sem.acquire()
        # print("YES2:",iter_id)

        inputs = batch
        labels = batch["labels"]
        total += batch["labels"].size(0)
        sub_model,sub_optimizer,sub_scheduler=train_variables.get_all_forward_element()
        inputs.to(get_device())
        intermediate = sub_model(inputs)
        train_variables.save_all_training_element(sub_model, sub_optimizer, sub_scheduler, inputs, intermediate[0])

        lr = sub_optimizer.param_groups[0]['lr']
        next_url = get_url_from_worker(get_stage_idx() + 1)
        loss_url = get_url_from_worker(get_worker_num() - 1)

        res = send_labels(loss_url, iter_id, labels)
        while res != "ok":
            time.sleep(10)
            res = send_labels(loss_url, iter_id, labels)
            print("send labels again")

        res = send_train_forward(next_url, iter_id, intermediate, 1, 0, ft_variables.get_training_term(),lr)
        while res != "ok":
            time.sleep(10)
            res = send_train_forward(next_url, iter_id, intermediate, 1, 0, ft_variables.get_training_term(), lr)
            print("send forward again")

    # get acc,loss on trainset
    while(id_now!=len(train_dataloader)):
        time.sleep(1)
    acc = correct / total * 100
    train_loss /= counter
    train_variables.weight_aggregate()
    return acc,train_loss

#@profile
def handle_backward_intermediate(req):
    # if req['term'] != ft_variables.get_training_term():
    #     print("Backwarding: Term mismatch!")
    #     return

    global correct, total, train_loss, counter, progress_bar

    iter_id = req['iter_id']
    # print("receive",iter_id)
    global id_now,lock

    lock.acquire()
    # print(iter_id,id_now)
    while (iter_id != id_now):
        lock.wait()

    id_now+=1

    sub_model,sub_optimizer,sub_scheduler,sub_input,sub_output=train_variables.get_all_backward_element()

    sub_optimizer.zero_grad()

    # print()
    # value0 = torch.tensor(req['data']).to(get_device())
    value0 = MNN_to_pytorch(req['data'])[0].to(get_device())
    # print(value0)
    sub_output.backward(value0)
    sub_optimizer.step()

    cur_correct = req['correct']
    cur_loss = req['loss']

    # cur_loss, cur_correct, num = train_step(batch, model, optimizer, iter_id)

    progress_bar.update(1)
    correct += cur_correct
    train_loss += cur_loss
    counter += 1

    sem = get_semaphore()
    sem.release()

    lock.notifyAll()
    lock.release()
    return
