import torch
from torch.utils.tensorboard import SummaryWriter

def write_computing_graph(input, model):
    writer = SummaryWriter(comment='Computing Graph',filename_suffix="computing_graph")

    writer.add_graph(model, input)  #模型及模型输入数据
    
    writer.close()

