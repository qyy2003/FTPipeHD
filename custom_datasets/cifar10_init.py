import torchvision.transforms as transforms
import torch
from global_variables.config import cfg
from torchvision.datasets import CIFAR10


def init_cifar10():
    """
        Initialize the CIFAR10 dataset and return the training set and test set
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    trainset = CIFAR10(cfg.data.train.dataset_path, transform=transform_train, train=True, download=True)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=2)

    testset = CIFAR10(cfg.data.val.dataset_path, transform=transform_test, train=False, download=True)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    return train_dataloader, test_dataloader