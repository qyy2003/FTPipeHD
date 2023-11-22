import torch
import torchvision
from torchsummary import summary

from models.mobilenetv2.model import MobileNetV2

def get_summary():
    model = MobileNetV2(width_mult=1)
    summary(model, input_size=(3, 32, 32))