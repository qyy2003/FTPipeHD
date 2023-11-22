import torch.nn as nn
import time
import numpy as np
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
import torch

class HARCNN(nn.Module):
    def __init__(self, args):
        super(HARCNN, self).__init__()

        self.n_chan = args["num_chan"] if args.get("num_chan") is not None else 9
        self.n_classes = args["num_class"] if args.get("num_class") is not None else 6
        self.total_layer = args["total_layer"] if args.get("total_layer") is not None else 5

        self.features = nn.Sequential(
            nn.Sequential(nn.Conv1d(self.n_chan, 64, kernel_size=3, stride=1), nn.ReLU()),
            nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, stride=1), nn.ReLU() ,nn.Dropout(p=0.6)),
            nn.MaxPool1d(kernel_size=2,stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Sequential(nn.Linear(3968, 100), nn.ReLU()),
            nn.Linear(100, self.n_classes)
        )
       
    def forward(self, data):
        inputs=data.get("inputs")
        labels=data.get("labels")
        # print(inputs)
        # print("---")
        # print(labels)
        batch_size = inputs.size(0)
        output = self.features(inputs)
        output = output.view((batch_size, -1))
        output = self.classifier(output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(output, labels)
            return output,loss
        else:
            return output

    def calculate_acc(self, output, labels):
        _, predicted = torch.max(output.data, -1)
        correct = (predicted == labels).sum().item()
        return correct

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


class SubHARCNN(nn.Module):
    def __init__(self, start, end, args: dict):
        super(SubHARCNN, self).__init__()

        self.n_chan = args["num_chan"] if args.get("num_chan") is not None else 9
        self.n_classes = args["num_class"] if args.get("num_class") is not None else 6
        total_layer = args["total_layer"] if args.get("total_layer") is not None else 9

        if end == -1:
            end = total_layer - 1

        feature_layers = [
            nn.Sequential(nn.Conv1d(self.n_chan, 64, kernel_size=3, stride=1), nn.ReLU()),
            nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, stride=1), nn.ReLU() ,nn.Dropout(p=0.6)),
            nn.MaxPool1d(kernel_size=2,stride=2),
        ]

        classifier_layers = [
            nn.Sequential(nn.Linear(3968, 100), nn.ReLU()),
            nn.Linear(100, self.n_classes)
        ]

        self.origin_features_len = len(feature_layers)
        self.origin_classifier_len = len(classifier_layers)
        
        features_ = []
        classifiers_ = []

        for i in range(total_layer):
            if i >= start and i <= end:
                if i < len(feature_layers):
                    features_.append(feature_layers[i])
                else:
                    classifiers_.append(classifier_layers[i - len(feature_layers)])
        
        self.features = nn.Sequential(*features_) if len(features_) > 0 else None
        self.classifier = nn.Sequential(*classifiers_) if len(classifiers_) > 0 else None
        self.is_classifier_start = start <= len(feature_layers) and end >= len(feature_layers)

    def forward(self, x):
        batch_size = x.size(0)
        if self.features is not None:
            x = self.features(x)
        
        if self.is_classifier_start:
            x = x.view((batch_size, -1))

        if self.classifier is not None:
            x = self.classifier(x)
        
        return x
