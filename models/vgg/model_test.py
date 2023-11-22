import torch
from model import VGG, SubVGG

def len_test():
    model = VGG({"type": "VGG11"})
    print(len(model.features))

    model = VGG({"type": "VGG13"})
    print(len(model.features))

    model = VGG({"type": "VGG16"})
    print(len(model.features))

    model = VGG({"type": "VGG19"})
    print(len(model.features))


def partition_test():
    model = VGG({"type": "VGG11"})
    
    sub_model = SubVGG(3, -1, {"type": "VGG11"})
    print(sub_model)


def forward_test():
    model = VGG({"type": "VGG13"})
    inputs = torch.rand(128, 3, 32, 32)
    output = model(inputs)
    print(output)


forward_test()