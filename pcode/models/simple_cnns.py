# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# from thop import clever_format, profile

__all__ = ["simple_cnn"]


def _decide_num_classes(dataset):
    if dataset == "cifar10" or dataset == "svhn":
        return 10
    elif dataset == "cifar100":
        return 100
    elif "imagenet" in dataset:
        return 1000
    elif "mnist" == dataset:
        return 10
    elif "fmnist" == dataset:
        return 10
    elif "femnist" == dataset:
        return 62
    else:
        raise NotImplementedError(f"this dataset ({dataset}) is not supported yet.")

class CNNMnist(nn.Module):
    def __init__(self, dataset, w_conv_bias=False, w_fc_bias=True):
        super(CNNMnist, self).__init__()

        # decide the num of classes.
        self.num_classes = _decide_num_classes(dataset)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, self.num_classes)

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['classifier.weight', 'classifier.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.classifier(x)
        return x


class CNNfemnist(nn.Module):
    def __init__(
        self, dataset, w_conv_bias=True, w_fc_bias=True, save_activations=True
    ):
        super(CNNfemnist, self).__init__()

        # decide the num of classes.
        self.num_classes = _decide_num_classes(dataset)

        # define layers.
        self.conv1 = nn.Conv2d(1, 32, 5, bias=w_conv_bias)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=w_conv_bias)
        self.fc1 = nn.Linear(1024, 2048, bias=w_fc_bias)
        self.classifier = nn.Linear(2048, self.num_classes, bias=w_fc_bias)

        # a placeholder for activations in the intermediate layers.
        self.save_activations = save_activations
        self.activations = None

    def forward(self, x):
        activation1 = self.conv1(x)
        x = self.pool(F.relu(activation1))

        activation2 = self.conv2(x)

        x = self.pool(F.relu(activation2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.classifier(x)

        if self.save_activations:
            self.activations = [activation1, activation2]
        return x


class CNNCifar(nn.Module):
    def __init__(
        self, dataset, w_conv_bias=True, w_fc_bias=True, save_activations=True
    ):
        super(CNNCifar, self).__init__()

        # decide the num of classes.
        self.num_classes = _decide_num_classes(dataset)

        # define layers.
        self.conv1 = nn.Conv2d(3, 6, 5, bias=w_conv_bias)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=w_conv_bias)
        self.fc1 = nn.Linear(16 * 5 * 5, 120, bias=w_fc_bias)
        self.fc2 = nn.Linear(120, 84, bias=w_fc_bias)
        self.classifier = nn.Linear(84, self.num_classes, bias=w_fc_bias)

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['classifier.weight', 'classifier.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

        # a placeholder for activations in the intermediate layers.
        self.save_activations = save_activations
        self.activations = None

    def forward(self, x):
        activation1 = self.conv1(x)
        x = self.pool(F.relu(activation1))

        activation2 = self.conv2(x)
        x = self.pool(F.relu(activation2))
        x = x.view(-1, 16 * 5 * 5)
        # x_f = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_f = x
        x = self.classifier(x)

        if self.save_activations:
            self.activations = [activation1, activation2, x_f]
        return x

class CNNCifar100(nn.Module):
    def __init__(
        self, dataset, w_conv_bias=True, w_fc_bias=True, save_activations=True
    ):
        super(CNNCifar100, self).__init__()

        # decide the num of classes.
        self.num_classes = _decide_num_classes(dataset)

        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.6)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, self.num_classes)

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['classifier.weight', 'classifier.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.drop((F.relu(self.fc2(x))))
        x = self.classifier(x)
        return x

def simple_cnn(conf):
    dataset = conf.data

    # if "cifar" in dataset or dataset == "svhn":
    if "cifar100" in dataset:
        return CNNCifar100(dataset, w_conv_bias=conf.w_conv_bias, w_fc_bias=conf.w_fc_bias)   
    elif "cifar10" in dataset:
        return CNNCifar(dataset, w_conv_bias=conf.w_conv_bias, w_fc_bias=conf.w_fc_bias) 
    elif "mnist" == dataset or "fmnist" == dataset:
        return CNNMnist(dataset, w_conv_bias=conf.w_conv_bias, w_fc_bias=conf.w_fc_bias)
    elif "femnist" == dataset:
        return CNNfemnist(
            dataset, w_conv_bias=conf.w_conv_bias, w_fc_bias=conf.w_fc_bias
        )
    else:
        raise NotImplementedError(f"not supported yet.")

    