import time
import torch
from memory_profiler import profile

import global_variables.fault_tolerance as ft_variables
from global_variables.common import get_stage_idx, get_worker_num, get_url_from_worker, get_workers, is_load_cp, log_message, set_stage_idx, update_workers, set_model_args, set_model_name,get_device
import global_variables.training as train_variables
from global_variables.record import update_forward_time, update_backward_time, get_forward_time, get_backward_time, reset_time
from utils.general import load_checkpoint, weight_aggregation, measure_bandwidth
from fault_tolerance.replication import replicate_weight
from utils.init import prepare_sub_model_optimizer_scheduler, init_sub_optimizer
from global_variables.config import cfg
from network.online import send_train_backward, send_train_forward
from flask_api.transfer import pytorch_to_MNN,MNN_to_pytorch
import time
import threading
from tensordict import TensorDict
import copy
forward_now=0
backward_now=0
total_len=0
lock=threading.Condition()
def copy_dic(dict):
    dict_new={}
    for key,val in dict.items():
        dict_new[key]=val
    return dict_new
#@profile
def handle_forward_intermediate(req):
    """
        Handle the forwarding intermediate result sending to this worker
    """
    # if int(req['term']) != ft_variables.get_training_term():
    #     print("Forwarding: Term mismatch!")
    #     return

    global forward_now,backward_now,lock
    lock.acquire()
    cur_stage = get_stage_idx()
    worker_num = get_worker_num()
    batch_diff = worker_num - cur_stage
    ret = {}
    iter_id = int(req['iter_id'])
    cur_model_idx = int(req['model_idx'])
    assert( cur_model_idx==cur_stage)
    while( forward_now!= iter_id or forward_now-backward_now >=batch_diff):
        lock.wait()
    forward_now+=1

    sub_model, sub_optimizer, sub_scheduler = train_variables.get_all_forward_element()

    # critical section
    start_time = time.time()
    sub_model.train()
    inputs = req['data']
    # print(inputs[0].requires_grad)
    # inputs[0].requires_grad=True
    labels=None
    if cur_model_idx == get_worker_num() - 1:
        labels=train_variables.get_label(iter_id)
    # inputs=TensorDict(inputs,batch_size=labels.size(0)).to(get_device())
    intermediate = sub_model(inputs,labels)

    # print(inputs[0])
    forward_time = time.time() - start_time

    # update_forward_time(time.time() - start_time)


    if cur_model_idx == get_worker_num() - 1:
        # If it is the final stage, backward it

        loss = intermediate[1]
        correct = sub_model.calculate_acc(intermediate[0], labels)
        # correct=1
        print("Backwarding Batch {} ｜ Current loss is {}...".format(iter_id, loss.item()))

        # warm-up phase, update the learning rate
        if req.get('lr') is not None:
            print("warm up phase, update learning rate...")
            train_variables.set_optimizer_lr(float(req['lr']))

        update_interval=1
        loss /= update_interval
            
        # Backward
        start_time = time.time()
        sub_optimizer.zero_grad()
        print(loss)
        loss.backward()
        sub_optimizer.step()
        backward_time = time.time() - start_time
        # update_backward_time(time.time() - start_time)

        # Send profiling time
        # if iter_id > 0 and (iter_id + 1) == train_variables.get_profiling_interval():
        #     print("Sending back profiling time ... ")
        #     ret['time'] = {}
        #     ret['time'][str(cur_model_idx)] = [get_forward_time() / train_variables.get_profiling_interval(), get_backward_time() / train_variables.get_profiling_interval()]
        #     #global_model.reset_time()
        #     if train_variables.get_profiling_interval() == 10:
        #         train_variables.update_profiling_interval(100)


        ret['loss'] = loss.item()
        ret['correct'] = correct
        # ret['data'] = inputs[0].grad.tolist()
        ret['data'] = pytorch_to_MNN([inputs[0].grad])
        ret['error_code'] = 0
        ret['iter_id'] = iter_id
        ret['version'] = req['version']
        # ret['term'] = req['term']
        
        backward_url = get_url_from_worker(cur_model_idx - 1)
        ret['model_idx'] = cur_model_idx - 1

        res = send_train_backward(backward_url, ret)
        if res != "ok" :
            print("Send train backward(): Neighbour connection fail")
        backward_now+=1
        lock.notifyAll()
        lock.release()

    else:

        train_variables.save_all_training_element(sub_model, sub_optimizer, sub_scheduler, inputs,
                                                  intermediate[0])

        next_url = get_url_from_worker(cur_model_idx + 1)

        del sub_model, sub_optimizer
        res = send_train_forward(next_url, iter_id, intermediate, cur_model_idx + 1, req['version'])
        # res = send_train_forward(next_url, iter_id, intermediate, cur_model_idx + 1, req['version'], req['term'])
        if res != "ok" :
            print("Send train forward(): Neighbour connection fail")

        lock.notifyAll()
        lock.release()

#@profile
def handle_backward_intermediate(req):
    """
        Handle the backwarding intermediate result sending to this worker
    """
    # if int(req['term']) != ft_variables.get_training_term():
    #     print("Backwarding: Term mismatch!")
    #     return

    global forward_now,backward_now,lock
    cur_stage = get_stage_idx()
    worker_num = get_worker_num()
    batch_diff = worker_num - cur_stage
    iter_id = int(req['iter_id'])
    cur_model_idx = int(req['model_idx'])
    print("# backward received:",iter_id,forward_now,backward_now,total_len)
    assert( cur_model_idx==cur_stage)
    lock.acquire()
    print("# backward received:",iter_id,forward_now,backward_now,total_len)
    while( backward_now!= iter_id or (forward_now-backward_now < batch_diff and total_len!=forward_now)):
        lock.wait()
    backward_now+=1


    ret = {}
    if req.get('loss') is not None:
        ret['loss'] = float(req['loss'])
        ret['correct'] = float(req['correct'])

    iter_id = int(req['iter_id'])
    cur_model_idx = int(req['model_idx'])


    sub_model, sub_optimizer, sub_scheduler, sub_input, sub_output = train_variables.get_all_backward_element()

    update_interval=1
    if (iter_id + 1) % update_interval == 0:
        sub_optimizer.zero_grad()

    print(iter_id)


    sub_optimizer.zero_grad()

    # print()
    value0 = MNN_to_pytorch(req['data'])[0].to(get_device())
    sub_output.backward(value0)
    sub_optimizer.step()

    if req.get('lr') is not None:
        print("warm up phase, update learning rate...")
        train_variables.set_optimizer_lr(float(req['lr']))

    # backward
    start_time = time.time()

    # print("version::", inter_result[0]._version)
    # value0 = torch.tensor(req['data'][0])
    # with torch.autograd.detect_anomaly():
    #     inter_result[0].backward(gradient=value0)
    # print(value0._version)
    # inter_result[0].backward()

    backward_time = time.time() - start_time
    # update_backward_time(time.time() - start_time)



    # ret['data'] = sub_input[0].grad.tolist()
    ret['data'] = pytorch_to_MNN([sub_input[0].grad])
    ret['error_code'] = 0
    ret['iter_id'] = req['iter_id']
    ret['version'] = req['version']
    ret['model_idx'] = cur_model_idx - 1
    # ret['term'] = req['term']

    # del sub_model, sub_optimizer
    # del inter_result, origin_input
    
    backward_url = get_url_from_worker(cur_model_idx - 1)
    res = send_train_backward(backward_url, ret)
    if(total_len==backward_now):
        train_variables.weight_aggregate()
    if res != "ok" :
        print("handle_backward_intermediate(): Neighbour connection fail")
    # train_variables.remove_inter_result(iter_id)
    lock.notifyAll()
    lock.release()

    
def init_epoch(epoch_id, lr, data_len):
    """
        Init the epoch for current training
    """
    train_variables.set_optimizer_lr(lr)
    global total_len,forward_now,backward_now
    total_len=int(data_len)
    forward_now = 0
    backward_now = 0
    # sub_optimizer = train_variables.get_sub_optimizer()
    #
    # # set the learning rate and reset the weight_pool
    # for param_group in sub_optimizer.param_groups:
    #     param_group['lr'] = float(lr)
    
    # weight_aggregation(get_stage_idx()) 
    # sub_optimizer.init_weight_pool()
    # train_variables.update_profiling_interval(10)
    reset_time()
    # sub_optimizer.batch_counter = 0
    ft_variables.set_start_iter_id(0)



def measure_neighbor_handler():
    """
        Measure the network of the worker given by index
    """
    cur_idx = get_stage_idx()
    print(str(get_stage_idx())+'+'+str(get_worker_num()))
    if cur_idx == get_worker_num() - 1:
        return "-1"
    target_url = get_url_from_worker(cur_idx + 1)
    # bw = measure_bandwidth(target_url)
    bw = 67602.88
    return bw


# def partition_handler(point):
#     """
#         Partition the model according to the partition point
#         ### deprecated method ###
#     """
#     print("Partitioning model and initializing optimizers ...")
#     create_sub_model(point)
#     if is_load_cp():
#         load_checkpoint()
#     train_variables.set_partition_point(point)
#     init_sub_optimizer()


def set_basic_info_handler(data):
    point = data['point']
    model_name = data['model_name']
    model_args = data['model_args']
    aggr_interval = data['aggr_interval']
    print(model_args)
    set_model_name(model_name)
    set_model_args(model_args)
    train_variables.set_aggregate_interval(aggr_interval)
    train_variables.set_partition_point(point)

    print("Create sub-model and initializing optimizers ...")
    prepare_sub_model_optimizer_scheduler(point)
    # if is_load_cp():
    #     load_checkpoint()