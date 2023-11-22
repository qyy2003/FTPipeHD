import torchvision.transforms as transforms
import torch
from global_variables.config import cfg
from torchvision.datasets import MNIST

def init_mnist():
    """
        Initialize the MNIST dataset and return the training set and test set
    """
    # 0.1307和0.3081是mnist数据集的均值和标准差，因为mnist数据值都是灰度图，所以图像的通道数只有一个，因此均值和标准差各一个
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = MNIST(cfg.data.train.dataset_path, transform=transform, train=True, download=True)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=2)

    testset = MNIST(cfg.data.val.dataset_path, transform=transform, train=False, download=True)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    return train_dataloader, test_dataloader