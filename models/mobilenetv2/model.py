import torch.nn as nn
import math
from copy import deepcopy
from memory_profiler import profile
import numpy as np
import time
import numpy as np
from torch.autograd import Variable

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True) # inplace is True in the original ver.
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, args: dict):
        super(MobileNetV2, self).__init__()

        n_class = args["n_class"] if args.get("n_class") is not None else 1000
        input_size = args["input_size"] if args.get("input_size") is not None else 224
        width_mult = args["width_mult"] if args.get("width_mult") is not None else 1.
        self.total_layer = args["total_layer"] if args.get("total_layer") is not None else 20

        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(nn.Linear(self.last_channel, n_class))

        self._initialize_weights()

    def forward(self, x):
        for block in self.features:
            x = block(x)
            
        # x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

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
            x = x.mean(3).mean(2)
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


class SubMobileNetV2(nn.Module):
    def __init__(self, start, end, args={}):
        super(SubMobileNetV2, self).__init__()

        n_class = args["n_class"] if args.get("n_class") is not None else 1000
        width_mult = args["width_mult"] if args.get("width_mult") is not None else 1.
        total_layer = args["total_layer"] if args.get("total_layer") is not None else 20

        last_channel = 1280
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel

        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        self.features = []
        self.classifier = None
        input_channel = 32
        block = InvertedResidual

        if end == -1:
            end = total_layer - 1
        
        if end == total_layer - 1:
            self.classifier = nn.Sequential(nn.Linear(self.last_channel, n_class))

        if start == 0:
            self.features.append(conv_bn(3, input_channel, 2))

        cur_layer = 1
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if cur_layer >= start and cur_layer <= end:
                    if i == 0:
                        self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                    else:
                        self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
                cur_layer += 1
                if cur_layer > end:
                    break
        
        if start < total_layer - 1 and end >= total_layer - 2:
            # building last several layers
            self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        
        self.features = nn.Sequential(*self.features)
        self._initialize_weights()
        self.origin_features_len = total_layer - 1
        self.origin_classifier_len = 1
    
    def forward(self, x):
        x = self.features(x)
        if self.classifier is not None:
            x = x.mean(3).mean(2)
            x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# if __name__ == '__main__':
    # model = MobileNetV2(width_mult=1)