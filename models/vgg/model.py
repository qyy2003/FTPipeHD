import torch
import torch.nn as nn
import time
import numpy as np
from torch.autograd import Variable


cfg = {
    'VGG11' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'VGG13' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'VGG16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'VGG19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'] 
}

vgg_len = {
    'VGG11' : 16,
    'VGG13' : 16,
    'VGG16' : 19,
    'VGG19' : 22
}

class VGG(nn.Module):
    def __init__(self, args: dict):
        super().__init__()
        
        type = args["type"] if args.get("type") is not None else "VGG11"
        num_class = args["num_class"] if args.get("num_class") is not None else 100
        self.total_layer = args["total_layer"] if args.get("total_layer") is not None else 16

        self.features = make_features(type, True)

        self.classifier = nn.Sequential(
            nn.Sequential(nn.Linear(512, 4096), nn.ReLU(inplace=True), nn.Dropout()),
            nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout()),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

    def profile_helper(self, x, rounds):
        forward_time = np.zeros(self.total_layer + 1)
        backward_time = np.zeros(self.total_layer + 1)

        batch_size = x.size(0)
        for i in range(rounds):
            outputs = []
            inputs = []
            print("Execution round {} start ...".format(i))
            # forward
            # feature
            for idx, module in enumerate(self.features):
                # detach from previous
                x = Variable(x.data, requires_grad=True)
                inputs.append(x)

                # compute output
                start_time = time.time()
                x = module(x)
                forward_time[idx] += (time.time() - start_time)

                outputs.append(x)
            
            x = Variable(x.data, requires_grad=True)
            inputs.append(x)
            start_time = time.time()
            x = x.view((batch_size, -1))
            forward_time[len(self.features)] += (time.time() - start_time)
            outputs.append(x)

            # classifier
            for idx, module in enumerate(self.classifier):
                # detach from previous
                x = Variable(x.data, requires_grad=True)
                inputs.append(x)

                # compute output
                start_time = time.time()
                x = module(x)
                forward_time[idx + len(self.features) + 1] += (time.time() - start_time)

                outputs.append(x)
        
            # backward
            g = x
            for i, output in reversed(list(enumerate(outputs))):
                if i == (len(outputs) - 1):
                    start_time = time.time()
                    output.backward(g)
                else:
                    start_time = time.time()
                    output.backward(inputs[i + 1].grad.data)
                
                backward_time[i] += (time.time() - start_time)
        
        forward_time /= rounds
        backward_time /= rounds
        return forward_time, backward_time


class SubVGG(nn.Module):
    def __init__(self, start, end, args: dict):
        super().__init__()

        type = args["type"] if args.get("type") is not None else "VGG11"
        num_class = args["num_class"] if args.get("num_class") is not None else 100
        total_layer = vgg_len[type]

        if end == -1:
            end = total_layer - 1
        
        self.origin_features_len = len(cfg[type])
        self.origin_classifier_len = total_layer - self.origin_features_len

        classifiers = [
            nn.Sequential(nn.Linear(512, 4096), nn.ReLU(inplace=True), nn.Dropout()),
            nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout()),
            nn.Linear(4096, num_class)
        ]

        if end < self.origin_features_len:
            self.features = make_sub_features(type, start, end, True)
            self.classifier = None
        elif start < self.origin_features_len and end >= self.origin_features_len:
            self.features = make_sub_features(type, start, self.origin_features_len - 1, True)
            classifier_ = []
            for i in range(self.origin_features_len, end + 1):
                classifier_.append(classifiers[i - self.origin_features_len])
            self.classifier = nn.Sequential(*classifier_)
        else:
            self.features = None
            classifier_ = []
            for i in range(start, end + 1):
                classifier_.append(classifiers[i - self.origin_features_len])
            self.classifier = nn.Sequential(*classifier_)
        
        print(self.features)
        print(self.classifier)
        self.is_classifier_start = start <= self.origin_features_len and end >= self.origin_features_len

    def forward(self, x):
        if self.features is not None:
            x = self.features(x)
        
        if self.is_classifier_start == True:
            x = x.view(x.size()[0], -1)

        if self.classifier is not None:
            x = self.classifier(x)

        return x
    

def make_features(type: str, batch_norm=False):
    layers = []

    input_channel = 3
    
    if cfg.get(type) is None:
        print("Type Error!")
        return 

    for l in cfg[type]:
        if l == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        layer = []
        layer.append(nn.Conv2d(input_channel, l, kernel_size=3, padding=1))

        if batch_norm:
            layer.append(nn.BatchNorm2d(l))

        layer.append(nn.ReLU(inplace=True))
        layers.append(nn.Sequential(*layer))  # * 是若干个无名参数

        input_channel = l

    return nn.Sequential(*layers)


def make_sub_features(type, start, end, batch_norm: False):
    layers = []
    
    if start == 0:
        input_channel = 3
    else:
        input_channel = -1
    
    if cfg.get(type) is None:
        print("Type Error!")
        return 

    cur_layer = 0
    for l in cfg[type]:
        if cur_layer < start:
            cur_layer += 1
            if l != 'M':
                input_channel = l
            continue
        
        cur_layer += 1
        if l == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue

        layer = []
        layer.append(nn.Conv2d(input_channel, l, kernel_size=3, padding=1))

        if batch_norm:
            layer.append(nn.BatchNorm2d(l))

        layer.append(nn.ReLU(inplace=True))
        layers.append(nn.Sequential(*layer))  # * 是若干个无名参数

        input_channel = l
        if cur_layer > end:
            break

    return nn.Sequential(*layers)
