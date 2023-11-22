import torch
class MNNtenor(object):
    def __init__(self,data):
        self.dim=data[1];
        self.tensor=torch.tensor(data[0]).reshape(self.dim)
        self.order=data[2];
        self.print()

    def export(self):
        return [self.tensor.flatten().numpy(),self.dim,self.order]

    def print(self):
        print(self.tensor.shape)
        print(self.dim)
        print(self.order)
        print("-------")