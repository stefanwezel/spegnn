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



# layer = EGNN_Sparse(feats_dim=10,
#                     pos_dim=2,
#                     orient_dim=2,
#                     # edge_attr_dim=4,
#                     m_dim=16,
#                     # fourier_features=4
#                     )

model = EGNN_Sparse_Network(
                    n_layers =3,
                    feats_dim=8,
                    # feats_dim=10,
                    pos_dim=2,
                    orient_dim=1,
                    edge_attr_dim=1, # for using rel_orient_dist as edge_attr
                    m_dim=16,
                    # embedding_nums=[5],
                    # embedding_dims=[10],
                    # edge_embedding_nums=[5], 
                    # edge_embedding_dims=[10],
                    )



training_set = GraphMNIST('.', is_train=True)
test_set = GraphMNIST('.', is_train=False)


batch_size = 1
train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
datum = next(iter(train_loader))
# print(layer.forward(datum.x, datum.edge_index, edge_attr=None))

# print(datum.x[.size())
# optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)
# optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
loss_function = torch.nn.CrossEntropyLoss()


for epoch in range(15):
    # training loop
    model.train()
    epoch_training_loss = 0
    for i, subsample in enumerate(train_loader):
        # data = data.to(device)
        optimizer.zero_grad()
        out_feats = model(subsample.x, subsample.edge_index, subsample.batch, None, bsize=batch_size)
        n_nodes = out_feats.size(0)
        scores = out_feats.view(batch_size,n_nodes,-1).mean(1)
        # scores = F.softmax(out_feats.view(batch_size,n_nodes,-1).mean(1), dim=1)
        loss = loss_function(scores,subsample.y)
        # if i % 250 == 0:
        #   print(loss.item())
        epoch_training_loss += loss.item()

        loss.backward()
        optimizer.step()

    optimizer.param_groups[0]['lr'] *= 0.9
    print(f"Training loss for epoch {epoch}: {epoch_training_loss / len(train_loader):.2f}")

    # evaluation loop
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for i, subsample in enumerate(test_loader):
            out_feats = model(subsample.x, subsample.edge_index, subsample.batch, None, bsize=batch_size)
            n_nodes = out_feats.size(0)
            scores = F.softmax(out_feats.view(batch_size,n_nodes,-1).mean(1), dim=1)
            _, predicted = torch.max(scores.data, 1)
            total_correct += (predicted == subsample.y).sum().item()

    accuracy = (100 * total_correct) / (len(test_loader)*batch_size)
    print(f"Evaluation accuracy for epoch {epoch}: {accuracy:.2f} percent.")
