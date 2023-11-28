
from custom_datasets.cifar100_init import init_cifar100
from custom_datasets.cifar10_init import init_cifar10
from custom_datasets.mnist_init import init_mnist
from custom_datasets.HAR.init import init_har
from custom_datasets.conll2003_init import init_conll2003
from custom_datasets.alpaca_init import init_alpaca
from custom_datasets.SQuAD_init import init_SQuAD
dataset_set = {
    "SQuAD":init_SQuAD,
    "Alpaca":init_alpaca,
    'conll2003':init_conll2003,
    'CIFAR10': init_cifar10,
    'CIFAR100': init_cifar100,
    'MNIST': init_mnist,
    'UCI_HAR': init_har
}

def init_dataset(dataset_name):
    assert dataset_set.get(dataset_name) is not None

    Init_dataset = dataset_set[dataset_name]
    train_dataloader, test_dataloader = Init_dataset()

    #TEST
    # import torch
    # train_size = 100
    # test_size = len(train_dataloader) - train_size
    # train_dataloader, test_dataset = torch.utils.data.random_split(train_dataloader, [train_size, test_size])

    return train_dataloader, test_dataloader