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


def plot_line(ax, center, slope, length=1):

    """ TODO """
    # if slope > 30:
    #     slope = 30
    # elif slope < -30:
    #     slope = -30
    # else:
    #     slope = slope
    slope = np.arctan(np.deg2rad(slope))
    if -0.01 < slope < 0.01:
        length = 0.5
    # elif (slope > 5) or (slope < -5):
    #     length = 0.5
    else:
        length=0.5#length

    b = center[1] - slope * center[0]
    pt1 = (center[0] - length, slope * (center[0] - length) + b)
    pt2 = (center[0] + length, slope * (center[0] + length) + b)

    # ax.plot((pt1[0], center[0]), (pt1[1], center[1]), color='red', linewidth=0.5)
    ax.plot((center[0], pt2[0]), (center[1], pt2[1]), color='red', linewidth=0.5)
# torch.manual_seed(0)


# model = Baseline_EGNN_Sparse_Network(
#                     n_layers =3,
#                     # feats_dim=8,
#                     feats_dim=10,
#                     pos_dim=2,
#                     update_coors=False,
#                     # edge_attr_dim=0, # for using rel_orient_dist as edge_attr
#                     m_dim=16,


model = EGNN_Sparse_Network(
                    n_layers =1,
                    # feats_dim=8,
                    # feats_dim=9,
                    feats_dim = 11,
                    pos_dim=2,
                    orient_dim=1,
                    update_coors=True,
                    # edge_attr_dim=1, # for using rel_orient_dist as edge_attr
                    edge_attr_dim=2, # updated version
                    m_dim=16,
                    )






training_set = GraphMNIST('.', is_train=True)
test_set = GraphMNIST('.', is_train=False)


batch_size = 1
train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
datum = next(iter(train_loader))
# print(layer.forward(datum.x, datum.edge_index, edge_attr=None))

# print(datum.x[.size())
# optimizer = torch.optim.Adam(model.parameters(),lr=1e-6)
# optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)
# optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
# optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)

loss_function = torch.nn.CrossEntropyLoss()





sample = next(iter(train_loader))
# print(sample.x.size())



coors = sample.x[:,:2]
orients = sample.x[:, 2:3]
# print(coors)


import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
coords = np.array(coors)*28
print(orients)
# # show the output of SLIC
# fig = plt.figure("Superpixels -- %d segments%")
# ax = fig.add_subplot(1, 1, 1)
ax.scatter(coords[:,0], coords[:,1])
# ax.imshow(mark_boundaries(image, segments))
# plt.axis("off")
# plot_line(orients)
for i in range(sample.x.size(0)):
    plot_line(ax, coords[i], orients[i])
plt.show()


# print(orients)

# for epoch in range(15):
#     # training loop
#     model.train()
#     epoch_training_loss = 0
#     for i, subsample in enumerate(train_loader):
#         # data = data.to(device)
#         optimizer.zero_grad()
#         out_feats = model(subsample.x, subsample.edge_index, subsample.batch, None, bsize=batch_size)
#         # n_nodes = out_feats.size(0)
#         # print(out_feats.size())
#         # print(n_nodes)
#         # print(out_feats.size())
#         scores = out_feats#.unsqueeze(0)
#         # print(out)
#         # scores = out_feats.view(batch_size,n_nodes,-1).mean(1)
#         loss = loss_function(scores,subsample.y)
#         # if i % 250 == 0:
#         #   print(loss.item())
#         epoch_training_loss += loss.item()

#         loss.backward()
#         optimizer.step()

#     # optimizer.param_groups[0]['lr'] *= 0.8
#     # optimizer.param_groups[0]['lr'] *= 0.9
#     optimizer.param_groups[0]['lr'] *= 0.95
#     # optimizer.param_groups[0]['lr'] *= 0.99
#     print(f"Training loss for epoch {epoch+1}: {epoch_training_loss / len(train_loader):.2f}")

#     # evaluation loop
#     model.eval()
#     total_correct = 0
#     with torch.no_grad():
#         for i, subsample in enumerate(test_loader):
#             out_feats = model(subsample.x, subsample.edge_index, subsample.batch, None, bsize=batch_size)
#             # n_nodes = out_feats.size(0)
#             # scores = F.softmax(out_feats.unsqueeze(0), dim=1)
#             scores = F.softmax(out_feats, dim=1)
#             # scores = F.softmax(out_feats.view(batch_size,n_nodes,-1).mean(1), dim=1)
#             _, predicted = torch.max(scores.data, 1)
#             total_correct += (predicted == subsample.y).sum().item()

#     accuracy = (100 * total_correct) / (len(test_loader)*batch_size)
#     print(f"Evaluation accuracy for epoch {epoch+1}: {accuracy:.2f} percent.")
