import numpy as np
import torch
from torchvision import datasets, transforms
from types import SimpleNamespace

import matplotlib.pyplot as plt
from skimage.data import astronaut
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.util import img_as_float
from skimage.future import graph
from skimage.measure import regionprops
from tqdm import tqdm

from models import E_GCL, EGNN, get_edges_batch
import pickle
import pandas as pd
# def collate(batch):
#     batch_feats = [b[0] for b in batch]
#     batch_coords = [b[1] for b in batch]
#     batch_targets = [b[2] for b in batch]
#     batch_feats = torch.nn.utils.rnn.pad_sequence(batch_feats,batch_first=True)
#     batch_coords = torch.nn.utils.rnn.pad_sequence(batch_coords,batch_first=True)
#     n_nodes = batch_feats.shape[1]
#     batch_size = batch_feats.shape[0]
#     edges, edge_attr = get_edges_batch(n_nodes, batch_size)
#     batch_coords = batch_coords.view(-1,2) / 28
#     batch_feats = batch_feats.view(batch_coords.shape[0],-1)
#     batch_targets = torch.LongTensor(batch_targets)
    
#     return [batch_feats,batch_coords,batch_targets,edges,n_nodes,batch_size]


# class SuperpixelDataset(torch.utils.data.Dataset):
#     """Whale dataset."""
#     def __init__(self,orig_dataset):
#         """
#         Args:
#             orig_dataset (Dataset): dataset
#         """
#         self.orig_dataset = orig_dataset
#     def __len__(self):
#         return len(self.orig_dataset)

#     def __getitem__(self, idx):
#         img,target = self.orig_dataset[idx]
#         img = np.float32(np.asarray(img[0,:,:]))/255
#         labels = slic(img, n_segments=25, compactness=0.5, sigma=0.1)
#         p = regionprops(labels+1,intensity_image=img)
#         g = graph.rag_mean_color(img, labels)
#         feats = []
#         coords = []
#         for node in g.nodes:
#             color = p[node]['mean_intensity']
#             invariants = p[node]['moments_hu']
#             center = torch.Tensor(p[node]['centroid']).unsqueeze(0)
#             feat = torch.cat([torch.Tensor([color]),torch.Tensor(invariants)]).unsqueeze(0)
#             feats.append(feat)
#             coords.append(center)
#         feats = torch.cat(feats,dim=0)
#         coords = torch.cat(coords,dim=0)
#         return (feats,coords,target)

# args = SimpleNamespace()
# args.no_cuda = False
# args.seed = 10
# args.n_epochs = 20
# args.batch_size = 128
# args.test_batch_size = 128

# use_cuda = not args.no_cuda and torch.cuda.is_available()

# torch.manual_seed(args.seed)

# device = torch.device("cuda" if use_cuda else "cpu")

# total_transform = transforms.Compose([
#                 #RandomReduceScale(0.3,1),
#                 #RandomRotate(-180,180),
#                 transforms.ToTensor()])
# total_transform_test = transforms.Compose([
#                 #RandomReduceScale(0.3,1),
#                 #RandomRotate(-180,180),
#                 transforms.ToTensor()])

# mnist_train = SuperpixelDataset(datasets.MNIST('../data', train=True, download=True,transform=total_transform))
# mnist_test = SuperpixelDataset(datasets.MNIST('../data', train=False, download=True,transform=total_transform_test))









def collate_graph(batch):
    """ TODO docstring"""
    # meta_data = [d['meta_data'].loc[d['meta_data']['is_darker']==1] for d in batch]

    batch_feats = [np.array(datum['feats']) for datum in batch]
    batch_coords = [np.array(datum['coords']) for datum in batch]


    # convert and pad locations
    batch_coords = torch.nn.utils.rnn.pad_sequence(
        [torch.from_numpy(coords) for coords in batch_coords],
        batch_first=True,
        )


    # pad features
    batch_feats = torch.nn.utils.rnn.pad_sequence(
        [torch.from_numpy(feats) for feats in batch_feats],
        batch_first=True,
        ).to(torch.float32)

    # obtain metadata and edges
    batch_size = batch_feats.size(0)
    n_nodes = batch_feats.size(1)
    feature_size = batch_feats.size(2)

    edges, _ = get_edges_batch(n_nodes, batch_size)

    return dict(
                feats = batch_feats.view(-1, feature_size).to(torch.float32),
                # coords = (batch_coords.view(-1,2) / 28).to(torch.float32),
                coords = batch_coords.view(-1,2).to(torch.float32),
                edges = edges,
                batch_size=batch_size,
                n_nodes=n_nodes,
                feature_size = feature_size,
                # img_paths=batch_img_paths,
                target=torch.LongTensor([b['targets'] for b in batch]),
        )


class GraphMNIST(torch.utils.data.Dataset):
    def __init__(self, data_root='./graph_mnist/',is_train=True):
        self.data_root = data_root
        self.is_train = is_train
        if self.is_train:
            with open(data_root + 'baseline_train_data.pickle', "rb") as fp:
                self.data_dict = pickle.load(fp)
        elif not self.is_train:
            with open(data_root + 'baseline_test_data.pickle', "rb") as fp:
                self.data_dict = pickle.load(fp)

        self.data = pd.DataFrame(self.data_dict)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ix):
        sample = self.data.iloc[ix]

        return dict(
            feats=sample['feats'],
            coords=sample['coords'],
            targets=sample['targets'],
            )



graph_mnist_train = GraphMNIST()
graph_mnist_test = GraphMNIST(is_train=False)



args = SimpleNamespace()
args.no_cuda = False
args.seed = 10
args.n_epochs = 20
args.batch_size = 128
args.test_batch_size = 1

use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(
    graph_mnist_train,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_graph,
    **kwargs)

test_loader = torch.utils.data.DataLoader(
    graph_mnist_test,
    batch_size=args.test_batch_size,
    shuffle=False,
    collate_fn=collate_graph,
    **kwargs)





sample = next(iter(train_loader))
print(sample.keys())


# print(sample['coords'].shape)



# i = np.random.randint(0, args.batch_size-1)
# print(i)
# coords = sample['coords'][i*sample['n_nodes']:i*sample['n_nodes']+sample['n_nodes']]
# plt.scatter(coords[:,0]*28,coords[:,1]*28)
# print(sample['target'][i])
# plt.show()




n_feat = 8

# Initialize EGNN
net = EGNN(in_node_nf=n_feat, hidden_nf=100, out_node_nf=10,in_edge_nf=0,attention=True, normalize=True,n_layers=6)
net = net.to(device)
# a,b=net(h, x, edges, edge_attr)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=1e-2)



subsample = next(iter(train_loader))


batch_feats = subsample['feats']
batch_coords = subsample['coords']
edges = subsample['edges']
n_nodes = subsample['n_nodes']
batch_size = subsample['batch_size']
target = subsample['target']




model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 2048, 1),
    torch.nn.ReLU(),
    torch.nn.LazyLinear(1024),
    torch.nn.ReLU(),
    torch.nn.LazyLinear(512),
    torch.nn.ReLU(),
    torch.nn.LazyLinear(256),
    torch.nn.ReLU(),
    torch.nn.LazyLinear(128),
    torch.nn.ReLU(),
    torch.nn.LazyLinear(64),
    torch.nn.ReLU(),
    torch.nn.LazyLinear(10),
    torch.nn.ReLU(),
    )



# print(model)

print(batch_coords.size())
print(batch_feats.size())


x = torch.cat((batch_coords, batch_feats),dim=1)
print(x.size())
scores = model(batch_coords.unsqueeze(0).unsqueeze(0))
# scores = scores.view(batch_size,n_nodes,-1).mean(1)
# print(scores.size())

# print(scores.size())







# for e in range(len(edges)):
#     edges[e] = edges[e].to(device)
# # edge_attr = edge_attr.to(device)
# feats, coords, target = batch_feats.to(device), batch_coords.to(device), target.to(device)
# out_feats,out_coords=net(feats, coords, edges, edge_attr=None)
# scores = out_feats.view(batch_size,n_nodes,-1).mean(1)










# print(test_loader)

# for epoch in range(args.n_epochs):
#     net.train()
#     pbar = tqdm(total=len(train_loader),position=0, leave=True)
#     for batch_idx, subsample in enumerate(train_loader):
#         batch_feats = subsample['feats']
#         batch_coords = subsample['coords']
#         edges = subsample['edges']
#         n_nodes = subsample['n_nodes']
#         batch_size = subsample['batch_size']
#         target = subsample['target']

#         for e in range(len(edges)):
#             edges[e] = edges[e].to(device)
#         # edge_attr = edge_attr.to(device)
#         feats, coords, target = batch_feats.to(device), batch_coords.to(device), target.to(device)
#         out_feats,out_coords=net(feats, coords, edges, edge_attr=None)
#         scores = out_feats.view(batch_size,n_nodes,-1).mean(1)
#         loss = loss_function(scores,target)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         pbar.set_description("Training loss: %f, Class: %f" % (loss.item(),(scores.max(1)[1]==target).float().mean().item() ) )
#         pbar.update()
#     pbar.close()
#     optimizer.param_groups[0]['lr'] *= 0.5
#     # Validation
#     #net.eval()
#     #pbar = tqdm(test_loader,position=0, leave=True)
#     #for batch_idx, (data, target) in enumerate(test_loader):
#     print("Starting evaluation...")
#     net.eval()
#     with torch.no_grad():
#         total_correct = 0
#         for _, test_sample in enumerate(test_loader):
#             feats = test_sample['feats']
#             coords = test_sample['coords']
#             edges = test_sample['edges']
#             n_nodes = test_sample['n_nodes']
#             batch_size = test_sample['batch_size']
#             target = test_sample['target']


#             for e in range(len(edges)):
#                 edges[e] = edges[e].to(device)
#             feats, coords, target = feats.to(device), coords.to(device), target.to(device)
#             out_feats,out_coords=net(feats, coords, edges, edge_attr=None)
#             scores = out_feats.view(batch_size,n_nodes,-1).mean(1)
#             _, predicted = torch.max(scores.data, 1)

#             total_correct += (predicted == target).sum().item()

#         accuracy = (100 * total_correct) / len(test_loader)
#         print(f"Evaluation accuracy for epoch {epoch}: {accuracy:.2f} percent.")