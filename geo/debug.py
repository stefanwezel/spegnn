import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (NNConv, global_mean_pool, graclus, max_pool,
                                max_pool_x)
from torch_geometric.utils import normalized_cut



from egnn_pytorch import *
from egnn_pytorch_geometric import *
from graph_mnist import GraphMNIST




path = 'data/'
transform = T.Cartesian(cat=False)

train_dataset = MNISTSuperpixels(path, True, transform=transform)[:6000]
# test_dataset = MNISTSuperpixels(path, False, transform=transform)[:1000]
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


sample = next(iter(train_loader))
print(sample.keys)
print(sample.x.size())
print(sample.edge_index.size())
print(sample.edge_attr.size())
# print(sample.x.size())


print(sample.y)
layer = EGNN_Sparse(feats_dim=1,
			        pos_dim=2,

                    # edge_attr_dim=4,
                    m_dim=16,
                    # fourier_features=4
                    )

feats = torch.randn(75, 1)
coors = torch.randn(75, 2)
x = torch.cat([coors, feats], dim=-1)
edge_idxs = (torch.rand(2, 1383) * 75).long()
print('-'*20)

print(x.dtype)
print(edge_idxs.dtype)
# print(layer.forward(x, edge_idxs, edge_attr=None).shape)
print('-'*20)







dataset = GraphMNIST('.')

train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
datum = next(iter(train_loader))
print(datum.x.dtype)
print(datum.edge_index.dtype)
print(layer.forward(datum.x, datum.edge_index, edge_attr=None).shape)