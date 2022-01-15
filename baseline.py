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

def collate(batch):
    batch_feats = [b[0] for b in batch]
    batch_coords = [b[1] for b in batch]
    batch_targets = [b[2] for b in batch]
    batch_feats = torch.nn.utils.rnn.pad_sequence(batch_feats,batch_first=True)
    batch_coords = torch.nn.utils.rnn.pad_sequence(batch_coords,batch_first=True)
    n_nodes = batch_feats.shape[1]
    batch_size = batch_feats.shape[0]
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)
    batch_coords = batch_coords.view(-1,2) / 28
    batch_feats = batch_feats.view(batch_coords.shape[0],-1)
    batch_targets = torch.LongTensor(batch_targets)
    
    return [batch_feats,batch_coords,batch_targets,edges,n_nodes,batch_size]


class SuperpixelDataset(torch.utils.data.Dataset):
    """Whale dataset."""
    def __init__(self,orig_dataset):
        """
        Args:
            orig_dataset (Dataset): dataset
        """
        self.orig_dataset = orig_dataset
    def __len__(self):
        return len(self.orig_dataset)

    def __getitem__(self, idx):
        img,target = self.orig_dataset[idx]
        img = np.float32(np.asarray(img[0,:,:]))/255
        labels = slic(img, n_segments=25, compactness=0.5, sigma=0.1)
        p = regionprops(labels+1,intensity_image=img)
        g = graph.rag_mean_color(img, labels)
        feats = []
        coords = []
        for node in g.nodes:
            color = p[node]['mean_intensity']
            invariants = p[node]['moments_hu']
            center = torch.Tensor(p[node]['centroid']).unsqueeze(0)
            feat = torch.cat([torch.Tensor([color]),torch.Tensor(invariants)]).unsqueeze(0)
            feats.append(feat)
            coords.append(center)
        feats = torch.cat(feats,dim=0)
        coords = torch.cat(coords,dim=0)
        return (feats,coords,target)

args = SimpleNamespace()
args.no_cuda = False
args.seed = 10
args.n_epochs = 20
args.batch_size = 128
args.test_batch_size = 128

use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

total_transform = transforms.Compose([
                #RandomReduceScale(0.3,1),
                #RandomRotate(-180,180),
                transforms.ToTensor()])
total_transform_test = transforms.Compose([
                #RandomReduceScale(0.3,1),
                #RandomRotate(-180,180),
                transforms.ToTensor()])

mnist_train = SuperpixelDataset(datasets.MNIST('../data', train=True, download=True,transform=total_transform))
mnist_test = SuperpixelDataset(datasets.MNIST('../data', train=False, download=True,transform=total_transform_test))

kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(mnist_train,
    batch_size=args.batch_size, shuffle=True, collate_fn=collate, **kwargs)
test_loader = torch.utils.data.DataLoader(mnist_test,
    batch_size=args.test_batch_size, shuffle=True,  collate_fn=collate, **kwargs)


n_feat = 8

# Initialize EGNN
net = EGNN(in_node_nf=n_feat, hidden_nf=100, out_node_nf=10,in_edge_nf=0,attention=True, normalize=True,n_layers=6)
net = net.to(device)
# a,b=net(h, x, edges, edge_attr)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=1e-2)

for epoch in range(args.n_epochs):
    pbar = tqdm(total=len(train_loader),position=0, leave=True)
    for batch_idx, (batch_feats, batch_coords, target, edges,n_nodes,batch_size) in enumerate(train_loader):

        for e in range(len(edges)):
            edges[e] = edges[e].to(device)
        #edge_attr = edge_attr.to(device)
        feats, coords, target = batch_feats.to(device), batch_coords.to(device), target.to(device)
        out_feats,out_coords=net(feats, coords, edges, edge_attr=None)
        scores = out_feats.view(batch_size,n_nodes,-1).mean(1)
        loss = loss_function(scores,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description("Training loss: %f, Class: %f" % (loss.item(),(scores.max(1)[1]==target).float().mean().item() ) )
        pbar.update()
    pbar.close()
    optimizer.param_groups[0]['lr'] *= 0.5
    # Validation
    #net.eval()
    #pbar = tqdm(test_loader,position=0, leave=True)
    #for batch_idx, (data, target) in enumerate(test_loader):