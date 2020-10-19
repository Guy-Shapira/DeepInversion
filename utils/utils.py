import torch
import torchvision
import torch.nn as nn
from torchvision.transforms import ToTensor
import numpy as np


def get_split_cifar100( batch_size=32, start=0, end=50):
    shuffle = False
    start_class = start
    end_class = end

    transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
    train = torchvision.datasets.CIFAR100('./data/', train=True, download=True, transform=transform_train)
    test = torchvision.datasets.CIFAR100('./data/', train=False, download=True, transform=transform_test)

    targets_train = torch.tensor(train.targets)
    target_train_idx = ((targets_train >= start_class) & (targets_train < end_class))

    targets_test = torch.tensor(test.targets)
    target_test_idx = ((targets_test >= start_class) & (targets_test < end_class))

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.dataset.Subset(train, np.where(target_train_idx == 1)[0]), batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(test, np.where(target_test_idx == 1)[0]),
                                              batch_size=batch_size)

    return train_loader, test_loader
