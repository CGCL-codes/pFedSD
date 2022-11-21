# -*- coding: utf-8 -*-
import torch.nn as nn

__all__ = ["mlp"]


# class MLP(nn.Module):
#     def __init__(self, dataset, num_layers, hidden_size, drop_rate):
#         super(MLP, self).__init__()
#         self.dataset = dataset

#         # init
#         self.num_layers = num_layers
#         self.num_classes = self._decide_num_classes()
#         input_size = self._decide_input_feature_size()

#         # define layers.
#         for i in range(1, self.num_layers + 1):
#             in_features = input_size if i == 1 else hidden_size
#             out_features = hidden_size

#             layer = nn.Sequential(
#                 nn.Linear(in_features, out_features),
#                 nn.BatchNorm1d(out_features),
#                 nn.ReLU(),
#                 nn.Dropout(p=drop_rate),
#             )
#             setattr(self, "layer{}".format(i), layer)

#         self.classifier = nn.Linear(hidden_size, self.num_classes, bias=False)

#     def _decide_num_classes(self):
#         if self.dataset == "cifar10":
#             return 10
#         elif self.dataset == "cifar100":
#             return 100

#     def _decide_input_feature_size(self):
#         if "cifar" in self.dataset:
#             return 32 * 32 * 3
#         elif "mnist" in self.dataset:
#             return 28 * 28
#         else:
#             raise NotImplementedError

#     def forward(self, x):
#         out = x.view(x.size(0), -1)

#         for i in range(1, self.num_layers + 1):
#             out = getattr(self, "layer{}".format(i))(out)
#         out = self.classifier(out)
#         return out

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden1 = nn.Linear(512, 256)
        self.layer_hidden2 = nn.Linear(256, 256)
        self.layer_hidden3 = nn.Linear(256, 128)
        self.layer_out = nn.Linear(128, dim_out)
        # self.softmax = nn.Softmax(dim=1)
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            ['layer_hidden2.weight', 'layer_hidden2.bias'],
                            ['layer_hidden3.weight', 'layer_hidden3.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)

        x = self.layer_hidden1(x)
        x = self.relu(x)

        x = self.layer_hidden2(x)
        x = self.relu(x)

        x = self.layer_hidden3(x)
        x = self.relu(x)

        x = self.layer_out(x)
        # return self.softmax(x)
        return x

def mlp(conf):
    # return MLP(
    #     dataset=conf.data,
    #     num_layers=conf.mlp_num_layers,
    #     hidden_size=conf.mlp_hidden_size,
    #     drop_rate=conf.drop_rate,
    # )
    # self.num_classes = self._decide_num_classes()
    return MLP(dim_in=784, dim_hidden=256, dim_out=10)
