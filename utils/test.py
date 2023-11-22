import torch
import torch.nn as nn
from global_variables.common import get_is_checkpoint, get_worker_num

from global_variables.config import cfg
from models.general_model import init_model
from models.mobilenetv2.model import MobileNetV2
from utils.general import weight_sync
import math
import time


def test_distribute(test_dataloader, epoch):
    """
        test model on test_dataloader under distributed environment
    """
    # 这里采取的方法是同步权重，在本地进行 test
    print("Synchronizing weight from other workers...")
    # model = init_model(cfg.model_name, cfg.model_args)
    sub_models = weight_sync()

    is_checkpoint = get_is_checkpoint()
    
    for i in range(get_worker_num()):
        sub_models[i].eval()
        if is_checkpoint:
            save_path = "./model_state/sub_model_{}_epoch_{}_{}.pkl".format(i, epoch, math.floor(time.time()))
            torch.save(sub_models[i].state_dict(), save_path)
    
    correct, total = 0, 0
    loss, counter = 0, 0
    criterion = nn.CrossEntropyLoss()

    print("Start test evaluation...")
    with torch.no_grad():
        for (images, labels) in test_dataloader:
            if cfg.data.name == 'MNIST':
                images = images.repeat(1, 3, 1, 1) 
            x = images
            for i in range(get_worker_num()):
                x = sub_models[i](x)
            
            outputs = x
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item()
            counter += 1

    del sub_models # before deletion, weight should be stored ?
    return loss / counter, correct / total * 100