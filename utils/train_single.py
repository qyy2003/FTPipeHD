from global_variables.config import cfg
from global_variables.common import get_model_name, log_message, get_model_args, get_is_checkpoint, get_device, \
    is_load_cp

from custom_datasets.general_init import init_dataset
from models.general_model import init_model

import copy
import time
import math
import torch
import torch.nn as nn
from tqdm.auto import tqdm

def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions

def train_single():
    """
        Train the dataset on single machine
    """
    log_message("Training mode set to Single Mode ...")
    device = get_device()
    train_dataloader, test_dataloader = init_dataset(cfg.data.name)
    log_message("Creating model {} ...".format(get_model_name()))

    model = init_model(get_model_name(), get_model_args())
    model=model.to(device)


    if is_load_cp():
        load_epoch = 0
        log_message("Loading trained model weights of epoch {}".format(load_epoch))
        load_path = "./model_state/model_single_train_epoch_{}.pkl".format(load_epoch)
        model.load_state_dict(torch.load(load_path))

    optimizer = init_optimizer_single(model)

    scheduler = init_scheduler_single(optimizer)
    log_message('Start formal training...')
    total_train_time = 0
    for epoch in range(0, cfg.schedule.total_epochs):
        lr = optimizer.param_groups[0]['lr']
        start_time = time.time()

        train_epoch(model, optimizer, lr, train_dataloader, test_dataloader, epoch)

        end_time = time.time()
        total_train_time += (end_time - start_time)
        log_message("Train phase | Epoch {}, Train time {}, Total time {}".format(epoch, end_time - start_time, total_train_time))
        scheduler.step()
    log_message("Train finish | Total training time {}".format(total_train_time))
    print("Train finish")


def train_epoch(model, optimizer, lr, train_dataloader, test_dataloader, epoch):
    correct, total = 0, 0
    train_loss, counter = 0, 0
    progress_bar = tqdm(range(len(train_dataloader)))
    for iter_id, batch in enumerate(train_dataloader):
        model.train()
        cur_loss, cur_correct,num = train_step(batch, model, optimizer, iter_id)

        total += num
        progress_bar.update(1)
        correct += cur_correct
        train_loss += cur_loss
        counter += 1

    # get acc,loss on trainset
    acc = correct / total * 100
    # train_loss /= counter

    # test
    val_loss, val_acc = test_single(model, test_dataloader)
    log_msg = 'Train phase: lr {}, iteration {}, epoch {}, loss: {:.4f}  val_loss: {:.4f}  acc: {:.4f}%' \
                ' val_acc: {:.4f}%'.format(lr, iter_id, epoch, train_loss,
                                        val_loss, acc, val_acc)

    log_message(log_msg)

    if get_is_checkpoint():
        log_message("Saving current model of epoch {}".format(epoch))
        save_path = "./model_state/model_single_train_epoch_{}_{}.pkl".format(epoch, math.floor(time.time()))
        torch.save(model.state_dict(), save_path)



def train_step(batch, model, optimizer, iter_id):
    """
        One step in train on a single machine
    """
    batch=batch.to(get_device())
    outputs,loss = model(batch)
    # print(batch.keys())
    correct=model.calculate_acc(outputs,batch["labels"])

    update_interval = 2
    loss = loss / update_interval

    loss.backward()

    if (iter_id + 1) % update_interval == 0:
        optimizer.step()
        optimizer.zero_grad()

    return loss, correct,batch["labels"].size(0)


def test_single(model, test_dataloader):
    """
        test model on test_dataloader on a single machine
    """
    model.eval()
    correct, total = 0, 0
    loss, counter = 0, 0

    criterion = nn.CrossEntropyLoss()
    progress_bar = tqdm(range(len(test_dataloader)))
    with torch.no_grad():
        for batch in test_dataloader:
            batch=batch.to(get_device())

            outputs, loss_one = model(batch)
            correct_one=model.calculate_acc(outputs,batch["labels"])
            correct += correct_one
            loss += loss_one
            counter += 1
            total+=batch["labels"].size(0)
            progress_bar.update(1)

    return loss / counter, correct / total *100


def init_optimizer_single(model):
    optimizer_cfg = copy.deepcopy(cfg.schedule.optimizer)
    name = optimizer_cfg.pop('name')
    Optimizer = getattr(torch.optim, name)
    optimizer = Optimizer(params=model.parameters(), **optimizer_cfg)
    return optimizer


def init_scheduler_single(optimizer):
    schedule_cfg = copy.deepcopy(cfg.schedule.lr_schedule)
    name = schedule_cfg.pop('name')
    Scheduler = getattr(torch.optim.lr_scheduler, name)
    lr_scheduler = Scheduler(optimizer=optimizer, **schedule_cfg)
    return lr_scheduler