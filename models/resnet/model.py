import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import time
from torch.autograd import Variable


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        m.bias.data.zero_()


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, bn=False, stride=1):
        super(ResBlock, self).__init__()
        self.bn = bn
        if bn:
            self.bn0 = nn.BatchNorm2d(in_planes)

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1)
        if bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1)
        self.shortcut = nn.Sequential()

        if stride > 1:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn0(x))
        else:
            out = F.relu(x)

        if self.bn:
            out = F.relu(self.bn1(self.conv1(out)))
        else:
            out = F.relu(self.conv1(out))

        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class BasicBlock(nn.Module):
    def __init__(self, input_channel=3):
        super(BasicBlock, self).__init__()
        self.net = [nn.Conv2d(input_channel, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)]
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class ResNet(nn.Module):
    def __init__(self, args: dict):
        super(ResNet, self).__init__()

        n_class = args["n_class"] if args.get("n_class") is not None else 1000
        self.total_layer = args["total_layer"] if args.get("total_layer") is not None else 20

        input_channel = 3

        self.features = []
        self.features += [nn.Conv2d(input_channel, 64, 3, 1, 1)]
        self.features += [nn.BatchNorm2d(64)]
        self.features += [nn.ReLU()]

        self.features += [nn.MaxPool2d(2)]
        
        self.features += [ResBlock(64, 64)]

        self.features += [ResBlock(64, 128, stride=2)]
        self.features += [ResBlock(128, 128)]
        self.features += [ResBlock(128, 256, stride=2)]
        self.features += [ResBlock(256, 512, stride=2)]

        self.features += [nn.AdaptiveMaxPool2d((1, 1))]

        self.classifier = nn.Sequential(nn.Linear(512, n_class))

        self.features = nn.Sequential(*self.features)

        self.features.apply(init_weights)
        self.classifier.apply(init_weights)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def profile_helper(self, x, rounds):
        forward_time = np.zeros(self.total_layer + 1)
        backward_time = np.zeros(self.total_layer + 1)

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
            x = x.view(x.size(0), -1) 
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
        
        print("Profiling execution finished ....")
        forward_time /= rounds
        backward_time /= rounds
        return forward_time, backward_time
    

class SubResNet(nn.Module):
    def __init__(self, start, end, args: dict):
        super(SubResNet, self).__init__()

        n_class = args["n_class"] if args.get("n_class") is not None else 1000
        total_layer = args["total_layer"] if args.get("total_layer") is not None else 20

        input_channel = 3

        self.classifier = None
        self.features = []
        self.maxpool = nn.Sequential(*[nn.AdaptiveMaxPool2d((1, 1))])

        if end == -1:
            end = total_layer - 1

        if end == total_layer - 1:
            self.classifier = nn.Sequential(nn.Linear(512, n_class))
            self.classifier.apply(init_weights)
            end = end - 1
        
        self.origin_features = [BasicBlock(input_channel), ResBlock(64, 64), ResBlock(64, 128, stride=2),
                                ResBlock(128, 128), ResBlock(128, 256, stride=2), ResBlock(256, 512, stride=2)]
       
        if start < total_layer - 1:
            self.features = self.origin_features[start : end + 1]

        self.features = nn.Sequential(*self.features)
        self.features.apply(init_weights) 

        self.origin_features_len = total_layer - 1
        self.origin_classifier_len = 1
    
    def forward(self, x):
        x = self.features(x)
        if self.classifier is not None:
            x = self.maxpool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        return x